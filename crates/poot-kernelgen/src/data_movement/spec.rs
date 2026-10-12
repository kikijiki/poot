//! What a data-movement request names: how the tensors are stored, and the index mapping from an output
//! element to the input element it copies.

use poot_kernel_ir::Ty;

use crate::KernelGenError;
use crate::elementwise::ELEMENTWISE_2D_WORKGROUP_SIZE;
use crate::request::Fold;

const G: &str = "data_movement";

/// How a tensor's logical elements sit in its buffer. Movement never decodes or re-encodes a payload:
/// every element moves as the bits its storage holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Storage {
    /// One logical element per buffer element. `carrier` is the buffer's element type (`F32`, `I32`, `F16`,
    /// `BF16`); a copy never converts, so a carrier names a bit width, not an interpretation.
    Elements {
        /// The buffer's element type.
        carrier: Ty,
    },
    /// `lane_bits`-wide lanes packed little-endian into `u32` words, element `c` of a row at bit
    /// `(c % lanes) * lane_bits` of word `c / lanes`, with every row (the last axis) padded to a whole
    /// word. The padding bits are unspecified, so a movement neither reads nor requires them.
    PackedRows {
        /// Width of one lane in bits: 8 or 16 (a 32-bit lane is one element, use `Elements`).
        lane_bits: u32,
    },
}

impl Storage {
    /// The narrowest packed-row lane the generator writes: one byte (E4M3FN).
    pub const BYTE_LANES: Storage = Storage::PackedRows { lane_bits: 8 };

    /// Logical elements one buffer unit holds: a buffer element for [`Storage::Elements`], a `u32` word for
    /// [`Storage::PackedRows`]. The unit is what one thread owns, so two threads never write one word.
    pub(super) fn lanes(&self) -> usize {
        match self {
            Storage::Elements { .. } => 1,
            Storage::PackedRows { lane_bits } => (32 / lane_bits) as usize,
        }
    }

    /// The type of one buffer element, as a kernel parameter's slice element.
    pub(super) fn unit_ty(&self) -> Ty {
        match self {
            Storage::Elements { carrier } => carrier.clone(),
            Storage::PackedRows { .. } => Ty::U32,
        }
    }

    /// Buffer units a tensor of `shape` occupies.
    pub fn units(&self, shape: &[usize]) -> usize {
        match self {
            Storage::Elements { .. } => shape.iter().product(),
            Storage::PackedRows { .. } => {
                let (rows, row_len) = rows_of(shape);
                rows.saturating_mul(row_len.div_ceil(self.lanes()))
            }
        }
    }

    fn check(&self) -> Result<(), KernelGenError> {
        match self {
            Storage::Elements { .. } => Ok(()),
            Storage::PackedRows { lane_bits: 8 | 16 } => Ok(()),
            Storage::PackedRows { lane_bits: 32 } => Err(KernelGenError::Unsupported {
                request: G,
                reason: "a 32-bit lane is one element: use Storage::Elements",
            }),
            Storage::PackedRows { .. } => Err(KernelGenError::Unsupported {
                request: G,
                reason: "packed-row lanes are 8 or 16 bits wide",
            }),
        }
    }
}

/// `shape` as `(rows, row_len)`: the last axis is the row, a scalar is one row of one element.
pub(super) fn rows_of(shape: &[usize]) -> (usize, usize) {
    match shape.split_last() {
        Some((&row_len, leading)) => (leading.iter().product(), row_len),
        None => (1, 1),
    }
}

/// Where a runtime or baked update slice starts along its axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UpdateIndex {
    /// Baked into the body.
    Static(usize),
    /// Read from the first element of an `I32` index buffer, so one captured dispatch serves every
    /// position.
    Runtime,
}

/// A window of a scatter's source units one dispatch writes: the unit `offset` the window starts at and the
/// `threads` it launches. Consecutive windows cover every source unit once, so a wide scatter splits into
/// dispatches that each stay under a device's watchdog budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitWindow {
    /// First output unit the window writes.
    pub offset: usize,
    /// Threads (one per unit) the window launches.
    pub threads: usize,
}

/// The index mapping of one movement. The output shape is [`MovementSpec::out_shape`]; every input shape a
/// variant carries is the logical shape, whatever the storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MovementOp {
    /// `out[idx] = in[idx permuted]`: output axis `d` reads input axis `perm[d]`.
    Transpose {
        /// The input's logical shape.
        in_shape: Vec<usize>,
        /// The input axis each output axis reads: a permutation of `0..rank`.
        perm: Vec<usize>,
    },
    /// `out = in[.., start..start + out_shape[axis], ..]` along `axis`.
    Slice {
        /// The input's logical shape.
        in_shape: Vec<usize>,
        /// The axis the slice is taken along.
        axis: usize,
        /// First input position on `axis` that the slice reads.
        start: usize,
    },
    /// `out` is `in` broadcast to `out_shape` (right-aligned; a broadcast axis reads the same element).
    Broadcast {
        /// The input's logical shape, right-aligned against `out_shape`.
        in_shape: Vec<usize>,
    },
    /// The flat logical order is unchanged and the row length changes. A `PackedRows` reshape that moves
    /// the last axis needs it because row padding is not part of the logical tensor.
    Reshape {
        /// The input's logical shape (same flat order as the output).
        in_shape: Vec<usize>,
    },
    /// `out[.., g, ..] = in[.., index[g], ..]` along `axis`; `index` is an `I32` buffer of `index_numel`
    /// elements (a scalar index is 1).
    Gather {
        /// The input's logical shape.
        in_shape: Vec<usize>,
        /// The axis the index selects along.
        axis: usize,
        /// Elements in the `I32` index buffer (`1` for a scalar index).
        index_numel: usize,
    },
    /// The inputs joined along `axis`; their extents on it sum to `out_shape[axis]`.
    Concat {
        /// The axis the inputs are joined along.
        axis: usize,
        /// The logical shape of each input, in join order.
        in_shapes: Vec<Vec<usize>>,
    },
    /// `out = operand` with `update` written at `[.., index..index + extent, ..]` on `axis`, where the
    /// operand has `out_shape` and the update has `update_shape`.
    UpdateSlice {
        /// The shape of the update tensor.
        update_shape: Vec<usize>,
        /// The axis the update is written along.
        axis: usize,
        /// Where the update starts on `axis`.
        index: UpdateIndex,
    },
    /// `out[slots[t], ..] = src[t, ..]` for each source row `t` with `slots[t] >= 0`: the paged-KV append,
    /// written in place. The kernel's output buffer is the pool itself (it already holds the base), so only
    /// the `src` rows and the `I32` `slots` buffer of `src_shape[0]` entries are inputs; `out_shape` is the
    /// pool's. One thread owns one source unit, so the launch is `O(src)`, not `O(out)`. `window`
    /// restricts the dispatch to part of the source units. The slots must be distinct (the graph's
    /// `ForwardSlotMap`): two rows naming one slot race.
    ScatterRows {
        /// The shape of the source rows' tensor (rows, then the trailing axes the output shares).
        src_shape: Vec<usize>,
        /// When set, the dispatch writes only this window of the output's units.
        window: Option<UnitWindow>,
    },
    /// `out[index[s], ..] = src[s, ..]` for an `I32` index that is a permutation of the output's rows. One
    /// thread owns one source element, so only [`Storage::Elements`] stores it.
    PermuteRows {
        /// The source tensor's logical shape; its rows are the index's length.
        src_shape: Vec<usize>,
    },
}

/// A data-movement kernel: the storage its tensors share, the output shape, the index mapping and how the
/// launch folds past the device's x-grid cap. `fold.numel` is [`MovementSpec::threads`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MovementSpec {
    /// How every tensor of the movement is stored.
    pub storage: Storage,
    /// The logical shape of the output.
    pub out_shape: Vec<usize>,
    /// The index mapping from output elements to input elements.
    pub op: MovementOp,
    /// How the launch folds past the device's x-grid cap; `fold.numel` is the thread count.
    pub fold: Fold,
}

impl MovementSpec {
    /// A request launched flat; [`Self::folded`] folds it onto a 2-D grid.
    pub fn new(storage: Storage, out_shape: Vec<usize>, op: MovementOp) -> Self {
        let mut spec = Self {
            storage,
            out_shape,
            op,
            fold: Fold {
                numel: 0,
                two_d: false,
                width: ELEMENTWISE_2D_WORKGROUP_SIZE,
            },
        };
        spec.fold.numel = spec.threads();
        spec
    }

    /// This request launched on a 2-D grid of [`ELEMENTWISE_2D_WORKGROUP_SIZE`]-lane workgroups when `two_d`.
    pub fn folded(mut self, two_d: bool) -> Self {
        self.fold.two_d = two_d;
        self
    }

    /// Threads the dispatch launches: one per output buffer unit (one per source unit for
    /// [`MovementOp::PermuteRows`] and [`MovementOp::ScatterRows`], one per window thread for a windowed
    /// scatter).
    pub fn threads(&self) -> usize {
        match &self.op {
            MovementOp::PermuteRows { src_shape, .. } => src_shape.iter().product(),
            MovementOp::ScatterRows {
                window: Some(window),
                ..
            } => window.threads,
            MovementOp::ScatterRows {
                src_shape,
                window: None,
            } => self.storage.units(src_shape),
            _ => self.storage.units(&self.out_shape),
        }
    }

    /// The generator's own preconditions on the request, checked before a statement exists.
    pub(super) fn validate(&self) -> Result<(), KernelGenError> {
        self.storage.check()?;
        if self.fold.numel != self.threads() {
            return Err(KernelGenError::NotEqualTo {
                generator: G,
                dim: "fold.numel".to_string(),
                value: self.fold.numel,
                expected: self.threads(),
            });
        }
        if let Some(axis) = self.out_shape.iter().position(|&extent| extent == 0) {
            return Err(KernelGenError::BelowMinimum {
                generator: G,
                what: format!("out_shape[{axis}]"),
                value: 0,
                min: 1,
            });
        }
        let rank = self.out_shape.len();
        let same_rank = |what: &str, shape: &[usize]| {
            if shape.len() == rank {
                return Ok(());
            }
            Err(KernelGenError::CountMismatch {
                generator: G,
                what: format!("{what} rank"),
                expected: rank,
                actual: shape.len(),
            })
        };
        let axis_in_range = |axis: usize, rank: usize| {
            if axis < rank {
                return Ok(());
            }
            Err(KernelGenError::AxisOutOfRange {
                generator: G,
                axis,
                rank,
            })
        };
        let nonempty = |what: &str, shape: &[usize]| match shape.iter().position(|&e| e == 0) {
            Some(axis) => Err(KernelGenError::BelowMinimum {
                generator: G,
                what: format!("{what}[{axis}]"),
                value: 0,
                min: 1,
            }),
            None => Ok(()),
        };
        match &self.op {
            MovementOp::Transpose { in_shape, perm } => {
                same_rank("in_shape", in_shape)?;
                same_rank("perm", perm)?;
                nonempty("in_shape", in_shape)?;
                let mut seen = vec![false; rank];
                for &axis in perm {
                    axis_in_range(axis, rank)?;
                    if std::mem::replace(&mut seen[axis], true) {
                        return Err(KernelGenError::Unsupported {
                            request: G,
                            reason: "a transpose permutation names an axis twice",
                        });
                    }
                }
                Ok(())
            }
            MovementOp::Slice {
                in_shape,
                axis,
                start,
            } => {
                same_rank("in_shape", in_shape)?;
                nonempty("in_shape", in_shape)?;
                axis_in_range(*axis, rank)?;
                let end = start.saturating_add(self.out_shape[*axis]);
                if end > in_shape[*axis] {
                    return Err(KernelGenError::ExceedsBound {
                        generator: G,
                        what: "slice end".to_string(),
                        value: end,
                        bound: in_shape[*axis],
                    });
                }
                Ok(())
            }
            MovementOp::Broadcast { in_shape } => {
                nonempty("in_shape", in_shape)?;
                if in_shape.len() > rank {
                    return Err(KernelGenError::ExceedsBound {
                        generator: G,
                        what: "in_shape rank".to_string(),
                        value: in_shape.len(),
                        bound: rank,
                    });
                }
                Ok(())
            }
            MovementOp::Reshape { in_shape } => {
                nonempty("in_shape", in_shape)?;
                let (elements, expected) = (
                    in_shape.iter().product::<usize>(),
                    self.out_shape.iter().product::<usize>(),
                );
                if elements != expected {
                    return Err(KernelGenError::NotEqualTo {
                        generator: G,
                        dim: "reshape element count".to_string(),
                        value: elements,
                        expected,
                    });
                }
                Ok(())
            }
            MovementOp::Gather {
                in_shape,
                axis,
                index_numel,
            } => {
                nonempty("in_shape", in_shape)?;
                axis_in_range(*axis, in_shape.len())?;
                if *index_numel == 0 {
                    return Err(KernelGenError::BelowMinimum {
                        generator: G,
                        what: "index_numel".to_string(),
                        value: 0,
                        min: 1,
                    });
                }
                Ok(())
            }
            MovementOp::Concat { axis, in_shapes } => {
                axis_in_range(*axis, rank)?;
                // One input is a copy, the degenerate concat.
                if in_shapes.is_empty() {
                    return Err(KernelGenError::BelowMinimum {
                        generator: G,
                        what: "concat inputs".to_string(),
                        value: 0,
                        min: 1,
                    });
                }
                let mut total = 0;
                for shape in in_shapes {
                    same_rank("concat input", shape)?;
                    nonempty("concat input", shape)?;
                    for d in (0..rank).filter(|&d| d != *axis) {
                        if shape[d] != self.out_shape[d] {
                            return Err(KernelGenError::ShapeMismatch {
                                generator: G,
                                what: format!("concat input off axis {axis}"),
                                a: shape.clone(),
                                b: self.out_shape.clone(),
                            });
                        }
                    }
                    total += shape[*axis];
                }
                if total != self.out_shape[*axis] {
                    return Err(KernelGenError::NotEqualTo {
                        generator: G,
                        dim: format!("concat extents on axis {axis}"),
                        value: total,
                        expected: self.out_shape[*axis],
                    });
                }
                Ok(())
            }
            MovementOp::UpdateSlice {
                update_shape,
                axis,
                index,
            } => {
                same_rank("update_shape", update_shape)?;
                nonempty("update_shape", update_shape)?;
                axis_in_range(*axis, rank)?;
                let extent = update_shape[*axis];
                if extent > self.out_shape[*axis] {
                    return Err(KernelGenError::ExceedsBound {
                        generator: G,
                        what: "update extent".to_string(),
                        value: extent,
                        bound: self.out_shape[*axis],
                    });
                }
                if (0..rank).any(|d| d != *axis && update_shape[d] != self.out_shape[d]) {
                    return Err(KernelGenError::ShapeMismatch {
                        generator: G,
                        what: format!("update off axis {axis}"),
                        a: update_shape.clone(),
                        b: self.out_shape.clone(),
                    });
                }
                match index {
                    UpdateIndex::Static(start) => {
                        let end = start.saturating_add(extent);
                        if end > self.out_shape[*axis] {
                            return Err(KernelGenError::ExceedsBound {
                                generator: G,
                                what: "update end".to_string(),
                                value: end,
                                bound: self.out_shape[*axis],
                            });
                        }
                        Ok(())
                    }
                    UpdateIndex::Runtime => Ok(()),
                }
            }
            MovementOp::ScatterRows { src_shape, window } => {
                same_rank("src_shape", src_shape)?;
                nonempty("src_shape", src_shape)?;
                // A thread copies one whole unit to the same offset of the named slot, which only holds
                // when a slot is a whole number of units. Packed lanes share a word along the last axis,
                // so a rank-1 pool (one lane per slot) has no such unit.
                if matches!(self.storage, Storage::PackedRows { .. }) && rank == 1 {
                    return Err(KernelGenError::Unsupported {
                        request: G,
                        reason: "a packed row scatter writes whole words, so a rank-1 pool (one lane per slot) has no unit to copy",
                    });
                }
                if rank == 0 || src_shape[1..] != self.out_shape[1..] {
                    return Err(KernelGenError::ShapeMismatch {
                        generator: G,
                        what: "scatter source off axis 0".to_string(),
                        a: src_shape.clone(),
                        b: self.out_shape.clone(),
                    });
                }
                let units = self.storage.units(src_shape);
                if let Some(window) = window
                    && window.offset >= units
                {
                    return Err(KernelGenError::ExceedsBound {
                        generator: G,
                        what: "scatter window offset".to_string(),
                        value: window.offset,
                        bound: units,
                    });
                }
                Ok(())
            }
            MovementOp::PermuteRows { src_shape } => {
                if !matches!(self.storage, Storage::Elements { .. }) {
                    return Err(KernelGenError::Unsupported {
                        request: G,
                        reason: "a permuted-row scatter writes one element per thread, so it needs element storage",
                    });
                }
                same_rank("src_shape", src_shape)?;
                nonempty("src_shape", src_shape)?;
                if rank == 0 || src_shape != &self.out_shape {
                    return Err(KernelGenError::ShapeMismatch {
                        generator: G,
                        what: "permuted-row scatter source".to_string(),
                        a: src_shape.clone(),
                        b: self.out_shape.clone(),
                    });
                }
                Ok(())
            }
        }
    }
}
