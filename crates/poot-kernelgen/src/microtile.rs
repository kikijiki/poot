//! The micro-tile a tiled schedule is written against (POOT-1396, `designs/prefill-microtile-1393.md` section 2):
//! the inner `C[m x n] += A[m x kappa] * B[kappa x n]` step over two operands staged in workgroup memory, and the
//! way a schedule reaches the accumulator's elements afterwards. The schedule owns the loop nest, the staging and
//! the grid; the micro-tile owns the accumulator registers, which lane holds which output (the owner map in
//! [`Geometry`]) and the arithmetic of one step.
//!
//! A schedule sees an accumulator through four operations ([`MicroTile::accumulator`], [`MicroTile::zero`],
//! [`MicroTile::step`], [`MicroTile::elements`]) and the elements through [`Elements`]; it never matches on the
//! variant it holds. The element view is the only way out of an accumulator: an epilogue, a mask, a softmax or a
//! store is written once against [`Elements`] and the owner map, never against registers.
//!
//! Only [`MicroTile::Scalar`] exists. The matrix-fragment implementation (`Fragment`, over the Kernel IR `Wmma*`
//! statements) joins the enum with the narrow-operand decision (option C of `designs/prefill-gap-1384.md`).
//!
//! The tiled attention schedule (POOT-1402) adds the per-row operations an online softmax needs, over these same
//! types: [`RowValues`] (one value per accumulator row a lane owns), [`MicroTile::scale_rows`] (the rescale of a
//! live accumulator), [`Elements::map`] and [`Elements::map_rows`] (score transform, mask, `exp`, `1 / l`),
//! [`Elements::row_reduce`] (a row's maximum or sum over the lane's own elements; the combine over the row's
//! owner lanes goes through workgroup memory, [`RowValues::publish`]), and [`Elements::stage`] (an element tile
//! written back to workgroup memory as the next product's operand).

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Constant, Local, Operand, Place, Rvalue, Terminator, Ty,
};

use crate::emit::{Emit, bin, copy, cu};
use crate::helpers::{Alloc, elem, guard};

/// The inner product step a tiled schedule runs, one implementation per kind of arithmetic unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MicroTile {
    /// Each lane accumulates `rows x cols` outputs in f32 registers; one step (kappa = 1) is `rows + cols` LDS
    /// reads and `rows * cols` FMAs, unrolled by the caller over its staged depth.
    Scalar { rows: LaneRows, cols: MicroCols },
}

/// The output rows a [`MicroTile::Scalar`] lane accumulates, built by the schedules that size a tiling (and test
/// fixtures).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaneRows(pub(crate) u32);

impl LaneRows {
    /// The rows per lane, at least 1.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The bare count, so a schedule's `Debug` (part of the planner's kernel key) reads `rows: 8`.
impl std::fmt::Debug for LaneRows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The output columns a [`MicroTile::Scalar`] lane accumulates.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MicroCols(u32);

impl MicroCols {
    /// `cols` columns per lane; a zero count fails the build of the constant that names it.
    pub(crate) const fn new(cols: u32) -> Self {
        assert!(cols > 0, "a micro-tile has at least one column");
        Self(cols)
    }

    /// The columns per lane, at least 1.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The bare count, so a schedule's `Debug` (part of the planner's kernel key) reads `cols: 4`.
impl std::fmt::Debug for MicroCols {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The lanes of one workgroup over its output tile, and the owner map: which tile row and column each of this
/// lane's accumulator elements is. Built once per body, in its setup block, by [`MicroTile::geometry`].
pub(crate) struct Geometry {
    /// The tile row of this lane's element row `i`.
    rows: Vec<Local>,
    /// The tile column of this lane's element column `j`.
    cols: Vec<Local>,
    /// This lane's column in the lane grid: which of a row's owner lanes it is.
    lane_col: Local,
    /// The lanes that own elements of one tile row (the grid's columns).
    row_owners: usize,
}

/// One value per accumulator row a lane owns, in the owner map's row order: a row's factor, bound or partial
/// reduction.
pub(crate) struct RowValues {
    values: Vec<Local>,
}

/// One operand tile in workgroup memory: element `(k, x)` of the staged depth `k` and the tile's row or column
/// `x` is word `k * stride + x` of LDS array `array` (`K`-major: adjacent lanes read adjacent words).
#[derive(Clone, Copy)]
pub(crate) struct Staged {
    pub(crate) array: u8,
    pub(crate) stride: usize,
}

/// A lane's accumulator registers, element `(i, j)` at `i * cols + j`. Built by [`MicroTile::accumulator`] and
/// left only through [`MicroTile::elements`].
pub(crate) struct Acc {
    cells: Vec<Local>,
}

/// An accumulator element's place in the output tile: its row and column, relative to the tile's origin.
#[derive(Clone, Copy)]
pub(crate) struct Coord {
    pub(crate) row: Local,
    pub(crate) col: Local,
}

/// The owned elements of an accumulator with their [`Coord`], in the owner map's order.
pub(crate) struct Elements {
    cells: Vec<(Coord, Local)>,
    /// Consecutive cells per owned row: cell `n` is in the lane's row `n / per_row`.
    per_row: usize,
}

/// Where one element is stored, as [`Elements::store`]'s caller computes it: the store happens only when `ok` is
/// nonzero, to the row-major output's `(row, col)`.
pub(crate) struct StoreAt {
    pub(crate) ok: Local,
    pub(crate) row: Local,
    pub(crate) col: Local,
}

impl MicroTile {
    /// The `(rows, columns)` of the output tile a grid of `lanes_m x lanes_n` lanes covers.
    pub(crate) const fn output_tile(self, lanes_m: usize, lanes_n: usize) -> (usize, usize) {
        match self {
            Self::Scalar { rows, cols } => (lanes_m * rows.0 as usize, lanes_n * cols.0 as usize),
        }
    }

    /// The owner map of lane `lane` in a grid of `lanes_m x lanes_n` lanes (lane `l` is grid row `l / lanes_n`,
    /// column `l % lanes_n`), emitted into `e`. A `Scalar` lane owns rows `lane_row + lanes_m * i` and columns
    /// `lane_col + lanes_n * j`: interleaved, so adjacent lanes read adjacent LDS words in [`Self::step`].
    pub(crate) fn geometry(
        self,
        e: &mut Emit,
        lane: Local,
        lanes_m: usize,
        lanes_n: usize,
    ) -> Geometry {
        match self {
            Self::Scalar { rows, cols } => {
                let lane_row = bin(e, Ty::Usize, BinOp::Div, copy(lane), cu(lanes_n));
                let lane_col = bin(e, Ty::Usize, BinOp::Rem, copy(lane), cu(lanes_n));
                let rows = (0..rows.0 as usize)
                    .map(|i| bin(e, Ty::Usize, BinOp::Add, copy(lane_row), cu(lanes_m * i)))
                    .collect();
                let cols = (0..cols.0 as usize)
                    .map(|j| bin(e, Ty::Usize, BinOp::Add, copy(lane_col), cu(lanes_n * j)))
                    .collect();
                Geometry {
                    rows,
                    cols,
                    lane_col,
                    row_owners: lanes_n,
                }
            }
        }
    }

    /// The accumulator registers one lane holds.
    pub(crate) const fn accumulators(self) -> usize {
        match self {
            Self::Scalar { rows, cols } => (rows.0 * cols.0) as usize,
        }
    }

    /// Declare a lane's accumulator registers. They hold nothing until [`Self::zero`].
    pub(crate) fn accumulator(self, al: &mut Alloc) -> Acc {
        match self {
            Self::Scalar { rows, cols } => Acc {
                cells: (0..rows.0 * cols.0)
                    .map(|_| al.add(Ty::F32, true))
                    .collect(),
            },
        }
    }

    /// `acc = 0`.
    pub(crate) fn zero(self, e: &mut Emit, acc: &mut Acc) {
        match self {
            Self::Scalar { .. } => {
                let zero = Operand::Const(Constant::F32(0.0));
                for &cell in &acc.cells {
                    e.assign(Place::local(cell), Rvalue::Use(zero.clone()));
                }
            }
        }
    }

    /// `acc += A[:, k] * B[k, :]` for staged depth `k` (a compile-time offset into both tiles), both operands read
    /// from LDS through `g`'s owner map.
    pub(crate) fn step(
        self,
        e: &mut Emit,
        g: &Geometry,
        acc: &mut Acc,
        a: Staged,
        b: Staged,
        k: usize,
    ) {
        match self {
            Self::Scalar { .. } => {
                let read = |e: &mut Emit, operand: Staged, at: Local| {
                    let idx = bin(e, Ty::Usize, BinOp::Add, cu(k * operand.stride), copy(at));
                    e.read_lds(Ty::F32, operand.array, copy(idx))
                };
                let a_vals: Vec<Local> = g.rows.iter().map(|&row| read(e, a, row)).collect();
                let b_vals: Vec<Local> = g.cols.iter().map(|&col| read(e, b, col)).collect();
                for (i, &a_val) in a_vals.iter().enumerate() {
                    for (j, &b_val) in b_vals.iter().enumerate() {
                        // A fused multiply-add ties each product to its accumulator: an unfused product is
                        // independent of the chain, so a backend that does not contract it (AMDGCN) computes
                        // every product of the unrolled steps ahead of the adds and spills them.
                        let cell = acc.cells[i * b_vals.len() + j];
                        e.assign(
                            Place::local(cell),
                            Rvalue::MulAdd(copy(a_val), copy(b_val), copy(cell)),
                        );
                    }
                }
            }
        }
    }

    /// The accumulator's elements with their tile coordinates, read in place.
    pub(crate) fn elements(self, g: &Geometry, acc: Acc) -> Elements {
        match self {
            Self::Scalar { .. } => {
                let coords = g
                    .rows
                    .iter()
                    .flat_map(|&row| g.cols.iter().map(move |&col| Coord { row, col }));
                Elements {
                    cells: coords.zip(acc.cells).collect(),
                    per_row: g.cols.len(),
                }
            }
        }
    }

    /// [`Self::step`] at a staged depth `k` known only at run time (a loop counter over a deep contraction).
    pub(crate) fn step_at(
        self,
        e: &mut Emit,
        g: &Geometry,
        acc: &mut Acc,
        a: Staged,
        b: Staged,
        k: Local,
    ) {
        match self {
            Self::Scalar { .. } => {
                let base = |e: &mut Emit, operand: Staged| {
                    bin(e, Ty::Usize, BinOp::Mul, copy(k), cu(operand.stride))
                };
                let (a_base, b_base) = (base(e, a), base(e, b));
                let read = |e: &mut Emit, operand: Staged, base: Local, at: Local| {
                    let idx = bin(e, Ty::Usize, BinOp::Add, copy(base), copy(at));
                    e.read_lds(Ty::F32, operand.array, copy(idx))
                };
                let a_vals: Vec<Local> =
                    g.rows.iter().map(|&row| read(e, a, a_base, row)).collect();
                let b_vals: Vec<Local> =
                    g.cols.iter().map(|&col| read(e, b, b_base, col)).collect();
                for (i, &a_val) in a_vals.iter().enumerate() {
                    for (j, &b_val) in b_vals.iter().enumerate() {
                        // Fused, as in `step`: an unfused product would be hoisted ahead of the adds and spilled.
                        let cell = acc.cells[i * b_vals.len() + j];
                        e.assign(
                            Place::local(cell),
                            Rvalue::MulAdd(copy(a_val), copy(b_val), copy(cell)),
                        );
                    }
                }
            }
        }
    }

    /// `acc[r, :] *= factor[r]` for every row the lane owns: the online-softmax rescale of a live accumulator.
    pub(crate) fn scale_rows(self, e: &mut Emit, acc: &mut Acc, factor: &RowValues) {
        match self {
            Self::Scalar { cols, .. } => {
                let per_row = cols.0 as usize;
                for (n, &cell) in acc.cells.iter().enumerate() {
                    let scaled = bin(
                        e,
                        Ty::F32,
                        BinOp::Mul,
                        copy(cell),
                        copy(factor.values[n / per_row]),
                    );
                    e.assign(Place::local(cell), Rvalue::Use(copy(scaled)));
                }
            }
        }
    }
}

impl Geometry {
    /// One value per owned row, `f(e, row)` of the row's tile coordinate.
    pub(crate) fn row_values(
        &self,
        e: &mut Emit,
        mut f: impl FnMut(&mut Emit, Local) -> Local,
    ) -> RowValues {
        RowValues {
            values: self.rows.iter().map(|&row| f(e, row)).collect(),
        }
    }

    /// The lanes that own elements of one tile row: the partials [`RowValues::publish`] writes per row.
    pub(crate) fn row_owners(&self) -> usize {
        self.row_owners
    }
}

impl RowValues {
    /// Write this lane's partial of each owned row to word `row * g.row_owners() + owner` of LDS array `array`
    /// (`owner` the lane's place among the row's owners), for a combine over the owners after a barrier.
    pub(crate) fn publish(&self, e: &mut Emit, g: &Geometry, array: u8) {
        for (&row, &value) in g.rows.iter().zip(&self.values) {
            let by_row = bin(e, Ty::Usize, BinOp::Mul, copy(row), cu(g.row_owners));
            let idx = bin(e, Ty::Usize, BinOp::Add, copy(by_row), copy(g.lane_col));
            e.write_lds(array, copy(idx), copy(value));
        }
    }
}

impl Elements {
    /// Replace every element in place by `f(e, coord, value)`. The cells keep their registers, so a map under a
    /// branch leaves them valid on both paths.
    pub(crate) fn map(
        &mut self,
        e: &mut Emit,
        mut f: impl FnMut(&mut Emit, Coord, Local) -> Local,
    ) {
        for &(coord, cell) in &self.cells {
            let next = f(e, coord, cell);
            e.assign(Place::local(cell), Rvalue::Use(copy(next)));
        }
    }

    /// [`Self::map`] with the element's row value from `rows`: `f(e, coord, row_value, value)`.
    pub(crate) fn map_rows(
        &mut self,
        e: &mut Emit,
        rows: &RowValues,
        mut f: impl FnMut(&mut Emit, Coord, Local, Local) -> Local,
    ) {
        for (n, &(coord, cell)) in self.cells.iter().enumerate() {
            let next = f(e, coord, rows.values[n / self.per_row], cell);
            e.assign(Place::local(cell), Rvalue::Use(copy(next)));
        }
    }

    /// Each owned row's `op` (an associative `f32` [`BinOp`]: `Max` or `Add`) over this lane's elements of it, in
    /// column order. The row's other owners hold the rest ([`RowValues::publish`]).
    pub(crate) fn row_reduce(&self, e: &mut Emit, op: BinOp) -> RowValues {
        RowValues {
            values: self
                .cells
                .chunks(self.per_row)
                .map(|row| {
                    row[1..].iter().fold(row[0].1, |acc, &(_, cell)| {
                        bin(e, Ty::F32, op, copy(acc), copy(cell))
                    })
                })
                .collect(),
        }
    }

    /// Write every element `(row, col)` to `into` as element `(k = col, x = row)`: word `col * stride + row`, the
    /// `A` operand of a product whose depth is this tile's columns (the `P` of attention's `P * V`).
    pub(crate) fn stage(&self, e: &mut Emit, into: Staged) {
        for &(coord, cell) in &self.cells {
            let by_col = bin(e, Ty::Usize, BinOp::Mul, copy(coord.col), cu(into.stride));
            let idx = bin(e, Ty::Usize, BinOp::Add, copy(by_col), copy(coord.row));
            e.write_lds(into.array, copy(idx), copy(cell));
        }
    }

    /// Hand each element and its coordinate to `f`, in the owner map's order, for a store whose guard needs
    /// control flow the straight-line [`Self::store`] chain does not build.
    pub(crate) fn for_each(self, mut f: impl FnMut(Coord, Local)) {
        for (coord, cell) in self.cells {
            f(coord, cell);
        }
    }
}

impl Elements {
    /// One guarded store per element into the row-major `out` of `row_stride` elements per row, as a chain of
    /// blocks starting at block `first`: block `first + 2n` evaluates element `n`'s [`StoreAt`] through `at`, and
    /// a nonzero `ok` enters block `first + 2n + 1`, which stores `value_at(e, row, out_idx, element)` to
    /// `out[out_idx]`, `out_idx = row * row_stride + col` (the hook a fused epilogue runs through: inside the
    /// element's guard, with the output row and the flat index its leaf reads broadcast from). Both continue to the next element's;
    /// the last continues to block `first + 2 * len`, which the caller builds.
    pub(crate) fn store(
        self,
        al: &mut Alloc,
        first: u32,
        out: Local,
        row_stride: Local,
        mut at: impl FnMut(&mut Emit, Coord) -> StoreAt,
        mut value_at: impl FnMut(&mut Emit, Local, Local, Local) -> Local,
    ) -> Vec<BasicBlock> {
        let mut blocks = Vec::with_capacity(2 * self.cells.len());
        for (n, (coord, value)) in self.cells.into_iter().enumerate() {
            let check = first + 2 * n as u32;
            let (write, next) = (check + 1, check + 2);

            let mut cond = Emit::new(al);
            let StoreAt { ok, row, col } = at(&mut cond, coord);
            blocks.push(BasicBlock {
                statements: cond.take(),
                terminator: guard(ok, next, write),
            });

            let mut store = Emit::new(al);
            let row_off = bin(
                &mut store,
                Ty::Usize,
                BinOp::Mul,
                copy(row),
                copy(row_stride),
            );
            let out_idx = bin(&mut store, Ty::Usize, BinOp::Add, copy(row_off), copy(col));
            let value = value_at(&mut store, row, out_idx, value);
            store.assign(elem(out, out_idx), Rvalue::Use(copy(value)));
            blocks.push(BasicBlock {
                statements: store.take(),
                terminator: Terminator::Goto {
                    target: BlockId { index: next },
                },
            });
        }
        blocks
    }
}
