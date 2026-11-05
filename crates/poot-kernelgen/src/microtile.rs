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
//! Two implementations exist. [`MicroTile::Scalar`] accumulates a register tile per lane with fused multiply-adds.
//! [`MicroTile::Fragment`] (POOT-1436, option C of `designs/prefill-gap-1384.md`) accumulates 16x16 matrix
//! fragments per wave over the Kernel IR `Wmma*` statements: narrow operands (bf16 or f16) in workgroup memory, an
//! f32 accumulator. The owners of a tile are lanes for the first and waves of [`FRAGMENT_LANES`] lanes for the
//! second ([`MicroTile::owner_lanes`]), so a grid of `owners_m x owners_n` is a grid of lanes or of waves.
//!
//! A fragment accumulator's elements are reached one of two ways ([`poot_target::FragmentLayout`]). A `Declared`
//! layout (AMDGCN gfx11) names which lane holds which element, so [`MicroTile::elements`] reads the registers in
//! place. An `Opaque` one (SPIR-V cooperative matrix, NVPTX `wmma`) does not, so [`MicroTile::spill`] stores each
//! fragment to a workgroup scratch array, the schedule runs a barrier, and [`MicroTile::elements`] reads the
//! elements back under the owner map this module chooses. Both present the same [`Elements`].
//!
//! The tiled attention schedule (POOT-1402) adds the per-row operations an online softmax needs, over these same
//! types: [`RowValues`] (one value per accumulator row a lane owns), [`MicroTile::scale_rows`] (the rescale of a
//! live accumulator), [`Elements::map`] and [`Elements::map_rows`] (score transform, mask, `exp`, `1 / l`),
//! [`Elements::row_reduce`] (a row's maximum or sum over the lane's own elements; the combine over the row's
//! owner lanes goes through workgroup memory, [`RowValues::publish`]), and [`Elements::stage`] (an element tile
//! written back to workgroup memory as the next product's operand).

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Constant, FRAGMENT_COLLECTIVE_LANES, FragmentMajor, FragmentRole,
    FragmentTy, Local, Operand, Place, Rvalue, Statement, Terminator, Ty, WmmaDtype, WmmaMat,
    WmmaShape, accumulator_map,
};
use poot_target::{FragmentDtype, FragmentLayout};

use crate::emit::{Emit, bin, copy, cu};
use crate::helpers::{Alloc, elem, guard};

/// Lanes in one fragment collective: the wave (or subgroup) a fragment is held across.
pub(crate) const FRAGMENT_LANES: usize = FRAGMENT_COLLECTIVE_LANES as usize;
/// The edge of a fragment: every config lowers 16x16x16.
const FRAGMENT_EDGE: usize = 16;

/// The inner product step a tiled schedule runs, one implementation per kind of arithmetic unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MicroTile {
    /// Each lane accumulates `rows x cols` outputs in f32 registers; one step (kappa = 1) is `rows + cols` LDS
    /// reads and `rows * cols` FMAs, unrolled by the caller over its staged depth.
    Scalar { rows: LaneRows, cols: MicroCols },
    /// Each wave accumulates `rows x cols` matrix fragments of 16x16 outputs; one step (kappa = 16) is `rows + cols`
    /// fragment loads from workgroup memory and `rows * cols` fragment multiplies.
    Fragment(FragmentTile),
}

/// The fragments one wave of a [`MicroTile::Fragment`] accumulates, and what its operands and accumulator are.
/// Built by the schedules that size a tiling (the planner's `RegisterTiling`) and carried by
/// [`crate::Schedule::FragmentTiled`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FragmentTile {
    /// The operand dtype of both multiplicands; the accumulator is f32.
    pub dtype: FragmentDtype,
    /// Whether the accumulator's elements are addressable in place.
    pub layout: FragmentLayout,
    /// Fragments along the output rows, at least 1.
    pub rows: u32,
    /// Fragments along the output columns, at least 1.
    pub cols: u32,
}

impl FragmentTile {
    fn wmma_dtype(self) -> WmmaDtype {
        match self.dtype {
            FragmentDtype::Bf16 => WmmaDtype::Bf16,
            FragmentDtype::F16 => WmmaDtype::F16,
        }
    }

    fn shape() -> WmmaShape {
        WmmaShape::M16N16K16
    }

    /// Elements one lane owns per fragment: `ELEMENTS_PER_LANE` rows of one column.
    const ELEMENTS: usize = accumulator_map::ELEMENTS_PER_LANE as usize;

    const fn fragments(self) -> usize {
        (self.rows * self.cols) as usize
    }
}

/// Which axis of a staged operand tile is contiguous in workgroup memory. A tile's element `(k, x)` (staged depth
/// `k`, the tile's row or column `x`) is word `k * stride + x` for [`StagedMajor::K`] (adjacent lanes adjacent in
/// `x`) and word `x * stride + k` for [`StagedMajor::X`] (a row's depth values adjacent: what a fragment lane
/// loads).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StagedMajor {
    K,
    X,
}

impl StagedMajor {
    /// The words between element `k` and `k + 1` of one `x`.
    pub(crate) const fn k_step(self, stride: usize) -> usize {
        match self {
            Self::K => stride,
            Self::X => 1,
        }
    }

    /// The words between element `x` and `x + 1` of one `k`.
    pub(crate) const fn x_step(self, stride: usize) -> usize {
        match self {
            Self::K => 1,
            Self::X => stride,
        }
    }
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
    /// Where this lane's wave sits in the tile, for a [`MicroTile::Fragment`].
    wave: Option<WavePlace>,
}

/// A fragment lane's place: its wave and the tile rows and columns the wave starts at (the origins of its
/// operand fragments), its lane within the wave's two half-waves and sixteen columns.
struct WavePlace {
    /// The wave's index in the workgroup.
    index: Local,
    /// The first tile row of the wave's fragments.
    row_origin: Local,
    /// The first tile column of the wave's fragments.
    col_origin: Local,
    /// `lane / 16` within the wave: which of the fragment's even and odd row sets this lane holds.
    half: Local,
    /// `lane % 16`: this lane's column of every fragment.
    column: Local,
}

/// One value per accumulator row a lane owns, in the owner map's row order: a row's factor, bound or partial
/// reduction.
pub(crate) struct RowValues {
    values: Vec<Local>,
}

/// One operand tile in workgroup memory: element `(k, x)` of the staged depth `k` and the tile's row or column
/// `x` is word `major.word(stride, k, x)` of LDS array `array`.
#[derive(Clone, Copy)]
pub(crate) struct Staged {
    pub(crate) array: u8,
    pub(crate) stride: usize,
    pub(crate) major: StagedMajor,
}

impl Staged {
    /// A `K`-major tile: word `k * stride + x`, adjacent lanes reading adjacent words.
    pub(crate) const fn k_major(array: u8, stride: usize) -> Self {
        Self {
            array,
            stride,
            major: StagedMajor::K,
        }
    }
}

/// A lane's accumulator: the f32 registers of a [`MicroTile::Scalar`] (element `(i, j)` at `i * cols + j`), or the
/// accumulator fragments of a [`MicroTile::Fragment`] (fragment `(i, j)` at `i * cols + j`). Built by
/// [`MicroTile::accumulator`] and left only through [`MicroTile::elements`].
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
    /// The lanes that make up one owner of the tile: one for a lane-owned [`Self::Scalar`] tile, a whole wave for a
    /// [`Self::Fragment`]. A workgroup of `owners_m x owners_n` owners runs that many times this many lanes.
    pub(crate) const fn owner_lanes(self) -> usize {
        match self {
            Self::Scalar { .. } => 1,
            Self::Fragment(_) => FRAGMENT_LANES,
        }
    }

    /// The lanes that own elements of one tile row in a grid of `owners_n` owner columns: one per column for a
    /// lane-owned tile, a fragment's 16 lanes of a row per wave column ([`accumulator_map`]).
    pub(crate) const fn row_owners(self, owners_n: usize) -> usize {
        match self {
            Self::Scalar { .. } => owners_n,
            Self::Fragment(_) => owners_n * accumulator_map::LANES_PER_ROW_SET as usize,
        }
    }

    /// The `(rows, columns)` of the output tile a grid of `owners_m x owners_n` owners covers.
    pub(crate) const fn output_tile(self, owners_m: usize, owners_n: usize) -> (usize, usize) {
        match self {
            Self::Scalar { rows, cols } => (owners_m * rows.0 as usize, owners_n * cols.0 as usize),
            Self::Fragment(tile) => (
                owners_m * tile.rows as usize * FRAGMENT_EDGE,
                owners_n * tile.cols as usize * FRAGMENT_EDGE,
            ),
        }
    }

    /// The staged depth one [`Self::step`] consumes: one value for a scalar FMA, a fragment's 16 for a multiply.
    pub(crate) const fn kappa(self) -> usize {
        match self {
            Self::Scalar { .. } => 1,
            Self::Fragment(_) => FRAGMENT_EDGE,
        }
    }

    /// The depth a schedule stages per barrier round unless it chooses its own: a whole number of [`Self::kappa`]
    /// steps, 16 values for a scalar tile and two fragment steps for a fragment tile. A fragment contraction sizes
    /// its round from its operands' global requests instead (`RegisterTiling`'s depth, POOT-1445); the tiled
    /// attention stages this one.
    pub(crate) const fn block_k(self) -> usize {
        match self {
            Self::Scalar { .. } => 16,
            Self::Fragment(_) => 2 * FRAGMENT_EDGE,
        }
    }

    /// The element type the staged tiles hold: f32 for a scalar tile, the fragment dtype for a fragment tile
    /// (the operands are already narrow, so workgroup memory halves and a fragment loads them as they are).
    pub(crate) const fn staged_ty(self) -> Ty {
        match self {
            Self::Scalar { .. } => Ty::F32,
            Self::Fragment(tile) => crate::precision::fragment_ty(tile.dtype),
        }
    }

    /// Bytes of one element of a staged tile.
    pub(crate) const fn staged_elem_bytes(self) -> usize {
        match self {
            Self::Scalar { .. } => 4,
            Self::Fragment(_) => 2,
        }
    }

    /// The value `x` (an f32 already rounded to the operand's declared precision) as a staged tile stores it: as is
    /// for an f32 tile, narrowed to the fragment dtype for a narrow one. The narrowing is exact
    /// ([`Rvalue::Narrow`]), because the rounding was done before it, on the integer bits ([`crate::precision`]):
    /// it must not round again (a second rounding costs a rounding sequence on a target with no native
    /// conversion), and must not be the rounding, which a driver compiler may drop from a narrow-and-widen pair.
    pub(crate) fn stage_value(self, e: &mut Emit, x: Local) -> Local {
        match self {
            Self::Scalar { .. } => x,
            Self::Fragment(_) => e.let_(
                self.staged_ty(),
                Rvalue::Narrow {
                    to: self.staged_ty(),
                    operand: copy(x),
                },
            ),
        }
    }

    /// How both operand tiles of a contraction are laid out in workgroup memory, whatever the weight's order in
    /// global memory: a scalar tile reads adjacent lanes' words, so `K`-major; a fragment lane loads 16
    /// consecutive depth values of its row (activation) or column (weight) in two 16-byte reads, so `X`-major.
    /// A `[K, N]` weight is transposed as it is staged (POOT-1445: staged `K`-major, a weight fragment was 16
    /// two-byte loads and the VALU that packs them).
    pub(crate) const fn staged_major(self) -> StagedMajor {
        match self {
            Self::Scalar { .. } => StagedMajor::K,
            Self::Fragment(_) => StagedMajor::X,
        }
    }

    /// Words a row of a staged tile of `edge` rows or columns is padded to, past its `depth` values (`X`-major,
    /// `depth` the round's) or its `edge` values (`K`-major), so cooperative stores and tile loads spread over the
    /// banks. A fragment's pad keeps every row 16-byte aligned for a vector load.
    pub(crate) const fn staged_stride(
        self,
        major: StagedMajor,
        edge: usize,
        depth: usize,
    ) -> usize {
        let (pad, extent) = match (self, major) {
            (Self::Scalar { .. }, _) => (1, edge),
            (Self::Fragment(_), StagedMajor::K) => (8, edge),
            (Self::Fragment(_), StagedMajor::X) => (8, depth),
        };
        extent + pad
    }

    /// Whether [`Self::spill`] and a barrier must separate the last [`Self::step`] from [`Self::elements`]: an
    /// `Opaque` fragment's elements are read back from workgroup memory.
    pub(crate) const fn needs_spill(self) -> bool {
        matches!(
            self,
            Self::Fragment(FragmentTile {
                layout: FragmentLayout::Opaque,
                ..
            })
        )
    }

    /// The f32 words of workgroup scratch a workgroup of `owners` owners needs for [`Self::spill`]: one 16x16 tile
    /// per fragment of each wave of an `Opaque` fragment tile, none otherwise.
    pub(crate) const fn scratch_words(self, owners: usize) -> usize {
        match self {
            Self::Fragment(tile) if self.needs_spill() => {
                owners * tile.rows as usize * tile.cols as usize * FRAGMENT_EDGE * FRAGMENT_EDGE
            }
            _ => 0,
        }
    }

    /// The owner map of lane `lane` in a grid of `owners_m x owners_n` owners (owner `o` is grid row `o / owners_n`,
    /// column `o % owners_n`; a scalar lane is its own owner, a fragment's owner is the wave `lane / 32`), emitted
    /// into `e`.
    ///
    /// - A `Scalar` lane owns rows `owner_row + owners_m * i` and columns `owner_col + owners_n * j`:
    ///   interleaved, so adjacent lanes read adjacent LDS words in [`Self::step`].
    /// - A `Fragment` lane owns, in each of its wave's fragments, columns `lane % 16` and the rows `2 e + lane / 16`
    ///   for `e` in `0..8` ([`accumulator_map`]): its owned row `fi * 8 + e` is the row `2 e + lane / 16` of
    ///   fragment row `fi`, and an owned column `fj` is column `lane % 16` of fragment column `fj`. The `Opaque`
    ///   layouts adopt the same map, so one element view serves both.
    pub(crate) fn geometry(
        self,
        e: &mut Emit,
        lane: Local,
        owners_m: usize,
        owners_n: usize,
    ) -> Geometry {
        match self {
            Self::Scalar { rows, cols } => {
                let lane_row = bin(e, Ty::Usize, BinOp::Div, copy(lane), cu(owners_n));
                let lane_col = bin(e, Ty::Usize, BinOp::Rem, copy(lane), cu(owners_n));
                let rows = (0..rows.0 as usize)
                    .map(|i| bin(e, Ty::Usize, BinOp::Add, copy(lane_row), cu(owners_m * i)))
                    .collect();
                let cols = (0..cols.0 as usize)
                    .map(|j| bin(e, Ty::Usize, BinOp::Add, copy(lane_col), cu(owners_n * j)))
                    .collect();
                Geometry {
                    rows,
                    cols,
                    lane_col,
                    row_owners: self.row_owners(owners_n),
                    wave: None,
                }
            }
            Self::Fragment(tile) => {
                let wave = bin(e, Ty::Usize, BinOp::Div, copy(lane), cu(FRAGMENT_LANES));
                let in_wave = bin(e, Ty::Usize, BinOp::Rem, copy(lane), cu(FRAGMENT_LANES));
                let wave_row = bin(e, Ty::Usize, BinOp::Div, copy(wave), cu(owners_n));
                let wave_col = bin(e, Ty::Usize, BinOp::Rem, copy(wave), cu(owners_n));
                let half = bin(
                    e,
                    Ty::Usize,
                    BinOp::Div,
                    copy(in_wave),
                    cu(accumulator_map::LANES_PER_ROW_SET as usize),
                );
                let column = bin(
                    e,
                    Ty::Usize,
                    BinOp::Rem,
                    copy(in_wave),
                    cu(accumulator_map::LANES_PER_ROW_SET as usize),
                );
                let row_origin = bin(
                    e,
                    Ty::Usize,
                    BinOp::Mul,
                    copy(wave_row),
                    cu(tile.rows as usize * FRAGMENT_EDGE),
                );
                let col_origin = bin(
                    e,
                    Ty::Usize,
                    BinOp::Mul,
                    copy(wave_col),
                    cu(tile.cols as usize * FRAGMENT_EDGE),
                );
                let row_base = bin(e, Ty::Usize, BinOp::Add, copy(row_origin), copy(half));
                let col_base = bin(e, Ty::Usize, BinOp::Add, copy(col_origin), copy(column));
                let rows = (0..tile.rows as usize)
                    .flat_map(|fi| (0..FragmentTile::ELEMENTS).map(move |ev| (fi, ev)))
                    .map(|(fi, ev)| {
                        let at = fi * FRAGMENT_EDGE + 2 * ev;
                        bin(e, Ty::Usize, BinOp::Add, copy(row_base), cu(at))
                    })
                    .collect();
                let cols = (0..tile.cols as usize)
                    .map(|fj| {
                        bin(
                            e,
                            Ty::Usize,
                            BinOp::Add,
                            copy(col_base),
                            cu(fj * FRAGMENT_EDGE),
                        )
                    })
                    .collect();
                let owner_col = bin(
                    e,
                    Ty::Usize,
                    BinOp::Mul,
                    copy(wave_col),
                    cu(accumulator_map::LANES_PER_ROW_SET as usize),
                );
                let lane_col = bin(e, Ty::Usize, BinOp::Add, copy(owner_col), copy(column));
                Geometry {
                    rows,
                    cols,
                    lane_col,
                    row_owners: self.row_owners(owners_n),
                    wave: Some(WavePlace {
                        index: wave,
                        row_origin,
                        col_origin,
                        half,
                        column,
                    }),
                }
            }
        }
    }

    /// The f32 accumulator elements one lane owns.
    pub(crate) const fn accumulators(self) -> usize {
        match self {
            Self::Scalar { rows, cols } => (rows.0 * cols.0) as usize,
            Self::Fragment(tile) => tile.fragments() * FragmentTile::ELEMENTS,
        }
    }

    /// Declare a lane's accumulator. It holds nothing until [`Self::zero`].
    pub(crate) fn accumulator(self, al: &mut Alloc) -> Acc {
        match self {
            Self::Scalar { rows, cols } => Acc {
                cells: (0..rows.0 * cols.0)
                    .map(|_| al.add(Ty::F32, true))
                    .collect(),
            },
            Self::Fragment(tile) => Acc {
                cells: (0..tile.fragments())
                    .map(|_| {
                        al.add(
                            Ty::Fragment(FragmentTy::accumulator(
                                tile.wmma_dtype(),
                                FragmentTile::shape(),
                            )),
                            false,
                        )
                    })
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
            Self::Fragment(tile) => {
                for &cell in &acc.cells {
                    e.push(Statement::WmmaZero {
                        dtype: tile.wmma_dtype(),
                        shape: FragmentTile::shape(),
                        dst: cell,
                    });
                }
            }
        }
    }

    /// `acc += A[:, k..k + kappa] * B[k..k + kappa, :]` for staged depth `k` (a compile-time offset into both tiles,
    /// a multiple of [`Self::kappa`]), both operands read from workgroup memory through `g`'s owner map.
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
                    debug_assert_eq!(operand.major, StagedMajor::K, "a scalar tile reads K-major");
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
            Self::Fragment(tile) => tile.step(e, g, acc, a, b, cu(k)),
        }
    }

    /// The accumulator's elements with their tile coordinates. A scalar accumulator is read in place; a `Declared`
    /// fragment's registers are read element by element under the lane map; an `Opaque` fragment's are read back
    /// from the `scratch` array [`Self::spill`] filled, after the schedule's barrier.
    pub(crate) fn elements(
        self,
        e: &mut Emit,
        g: &Geometry,
        acc: Acc,
        scratch: Option<u8>,
    ) -> Elements {
        let coords = || {
            g.rows
                .iter()
                .flat_map(|&row| g.cols.iter().map(move |&col| Coord { row, col }))
                .collect::<Vec<_>>()
        };
        match self {
            Self::Scalar { .. } => Elements {
                cells: coords().into_iter().zip(acc.cells).collect(),
                per_row: g.cols.len(),
            },
            Self::Fragment(tile) => {
                let wave = g.wave.as_ref().expect("a fragment geometry has a wave");
                let mut values = Vec::with_capacity(tile.fragments() * FragmentTile::ELEMENTS);
                for fi in 0..tile.rows as usize {
                    for ev in 0..FragmentTile::ELEMENTS {
                        for fj in 0..tile.cols as usize {
                            let fragment = acc.cells[fi * tile.cols as usize + fj];
                            values.push(match tile.layout {
                                FragmentLayout::Declared(_) => {
                                    let value = e.fresh(Ty::F32, true);
                                    e.push(Statement::WmmaElemRead {
                                        src: fragment,
                                        idx: ev as u8,
                                        dst: value,
                                    });
                                    value
                                }
                                FragmentLayout::Opaque => {
                                    let array = scratch.expect(
                                        "an opaque fragment reads back from a scratch array",
                                    );
                                    let idx = tile.scratch_index(e, wave, fi, fj, ev);
                                    let value = e.fresh(Ty::F32, true);
                                    e.assign(
                                        Place::local(value),
                                        Rvalue::WorkgroupLocalRead {
                                            idx: copy(idx),
                                            array,
                                        },
                                    );
                                    value
                                }
                            });
                        }
                    }
                }
                Elements {
                    cells: coords().into_iter().zip(values).collect(),
                    per_row: g.cols.len(),
                }
            }
        }
    }

    /// Store every fragment of an `Opaque` accumulator to its tile of the f32 workgroup `scratch` array
    /// ([`Self::scratch_words`]): the wave's fragments at `((wave * rows + i) * cols + j) * 256`, row-major. The
    /// schedule runs a barrier before [`Self::elements`] reads them back. Nothing for any other micro-tile.
    pub(crate) fn spill(self, e: &mut Emit, g: &Geometry, acc: &Acc, scratch: Option<u8>) {
        let Self::Fragment(tile) = self else {
            return;
        };
        if !self.needs_spill() {
            return;
        }
        let wave = g.wave.as_ref().expect("a fragment geometry has a wave");
        let array = scratch.expect("an opaque fragment spills to a scratch array");
        for fi in 0..tile.rows as usize {
            for fj in 0..tile.cols as usize {
                let base = tile.scratch_base(e, wave, fi, fj);
                e.push(Statement::WmmaStoreLds {
                    dtype: tile.wmma_dtype(),
                    shape: FragmentTile::shape(),
                    array,
                    base: copy(base),
                    major: FragmentMajor::Row,
                    stride: FRAGMENT_EDGE as u32,
                    src: acc.cells[fi * tile.cols as usize + fj],
                });
            }
        }
    }

    /// `scratch[r, :] *= factor[r]` for every element of the `Opaque` accumulator [`Self::spill`] stored, each lane
    /// scaling the elements it owns under the owner map (a partition of the tile, so no two lanes touch a word).
    /// The schedule runs a barrier before it (every fragment is stored) and another before [`Self::reload`].
    pub(crate) fn scale_spilled(
        self,
        e: &mut Emit,
        g: &Geometry,
        factor: &RowValues,
        scratch: Option<u8>,
    ) {
        let Self::Fragment(tile) = self else {
            unreachable!("only a fragment tile is spilled");
        };
        assert!(self.needs_spill(), "a declared fragment scales in place");
        let wave = g.wave.as_ref().expect("a fragment geometry has a wave");
        let array = scratch.expect("an opaque fragment spills to a scratch array");
        for fi in 0..tile.rows as usize {
            for fj in 0..tile.cols as usize {
                for ev in 0..FragmentTile::ELEMENTS {
                    let idx = tile.scratch_index(e, wave, fi, fj, ev);
                    let value = e.read_lds(Ty::F32, array, copy(idx));
                    let scaled = bin(
                        e,
                        Ty::F32,
                        BinOp::Mul,
                        copy(value),
                        copy(factor.values[fi * FragmentTile::ELEMENTS + ev]),
                    );
                    e.write_lds(array, copy(idx), copy(scaled));
                }
            }
        }
    }

    /// Load every fragment of an `Opaque` accumulator back from the `scratch` array [`Self::spill`] filled (and
    /// [`Self::scale_spilled`] scaled), after the schedule's barrier: the accumulator is live again for the next
    /// [`Self::step`].
    pub(crate) fn reload(self, e: &mut Emit, g: &Geometry, acc: &mut Acc, scratch: Option<u8>) {
        let Self::Fragment(tile) = self else {
            unreachable!("only a fragment tile is spilled");
        };
        assert!(self.needs_spill(), "a declared fragment is never reloaded");
        let wave = g.wave.as_ref().expect("a fragment geometry has a wave");
        let array = scratch.expect("an opaque fragment spills to a scratch array");
        for fi in 0..tile.rows as usize {
            for fj in 0..tile.cols as usize {
                let base = tile.scratch_base(e, wave, fi, fj);
                e.push(Statement::WmmaLoadLds {
                    role: FragmentRole::Accumulator,
                    dtype: tile.wmma_dtype(),
                    shape: FragmentTile::shape(),
                    array,
                    base: copy(base),
                    major: FragmentMajor::Row,
                    stride: FRAGMENT_EDGE as u32,
                    dst: acc.cells[fi * tile.cols as usize + fj],
                });
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
            Self::Fragment(tile) => tile.step(e, g, acc, a, b, copy(k)),
        }
    }

    /// `acc[r, :] *= factor[r]` for every row the lane owns: the online-softmax rescale of a live accumulator.
    ///
    /// An `Opaque` fragment has no element access to rescale in place. Its rescale is [`Self::spill`], a
    /// barrier, [`Self::scale_spilled`], a barrier and [`Self::reload`], which the schedule sequences because
    /// the barriers are its blocks; this method is for a micro-tile that does not [`Self::needs_spill`].
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
            Self::Fragment(tile) => {
                assert!(
                    matches!(tile.layout, FragmentLayout::Declared(_)),
                    "an opaque fragment has no in-place rescale"
                );
                for fi in 0..tile.rows as usize {
                    for fj in 0..tile.cols as usize {
                        let fragment = acc.cells[fi * tile.cols as usize + fj];
                        for ev in 0..FragmentTile::ELEMENTS {
                            let value = e.fresh(Ty::F32, false);
                            e.push(Statement::WmmaElemRead {
                                src: fragment,
                                idx: ev as u8,
                                dst: value,
                            });
                            let scaled = bin(
                                e,
                                Ty::F32,
                                BinOp::Mul,
                                copy(value),
                                copy(factor.values[fi * FragmentTile::ELEMENTS + ev]),
                            );
                            e.push(Statement::WmmaElemWrite {
                                dst: fragment,
                                idx: ev as u8,
                                value: copy(scaled),
                            });
                        }
                    }
                }
            }
        }
    }
}

impl FragmentTile {
    /// The first word of fragment `(fi, fj)` of this lane's wave in the f32 scratch array an `Opaque` spill fills.
    fn scratch_base(self, e: &mut Emit, wave: &WavePlace, fi: usize, fj: usize) -> Local {
        let within = (fi * self.cols as usize + fj) * FRAGMENT_EDGE * FRAGMENT_EDGE;
        let wave_words = self.fragments() * FRAGMENT_EDGE * FRAGMENT_EDGE;
        let by_wave = bin(e, Ty::Usize, BinOp::Mul, copy(wave.index), cu(wave_words));
        bin(e, Ty::Usize, BinOp::Add, copy(by_wave), cu(within))
    }

    /// The scratch word of this lane's element `ev` of fragment `(fi, fj)` ([`Self::scratch_base`]): row
    /// `2 ev + lane / 16`, column `lane % 16` of the row-major 16x16 tile, the owner map of [`MicroTile::geometry`].
    fn scratch_index(
        self,
        e: &mut Emit,
        wave: &WavePlace,
        fi: usize,
        fj: usize,
        ev: usize,
    ) -> Local {
        let base = self.scratch_base(e, wave, fi, fj);
        let row = bin(e, Ty::Usize, BinOp::Add, copy(wave.half), cu(2 * ev));
        let by_row = bin(e, Ty::Usize, BinOp::Mul, copy(row), cu(FRAGMENT_EDGE));
        let at = bin(e, Ty::Usize, BinOp::Add, copy(by_row), copy(wave.column));
        bin(e, Ty::Usize, BinOp::Add, copy(base), copy(at))
    }

    /// `acc[i][j] += A[i] * B[j]` for the fragments of staged depth `k` (a multiple of 16): `rows` operand fragments
    /// of A and `cols` of B loaded from workgroup memory at the wave's origins, then one multiply per accumulator.
    fn step(self, e: &mut Emit, g: &Geometry, acc: &mut Acc, a: Staged, b: Staged, k: Operand) {
        let wave = g.wave.as_ref().expect("a fragment geometry has a wave");
        let a_frags = self.load(e, WmmaMat::A, a, wave.row_origin, k.clone());
        let b_frags = self.load(e, WmmaMat::B, b, wave.col_origin, k);
        self.multiply(e, acc, &a_frags, &b_frags);
    }

    /// The wave's operand fragments of `mat` at staged depth `k` of `operand`: one per fragment row (A) or column
    /// (B), from the wave's `origin`.
    fn load(
        self,
        e: &mut Emit,
        mat: WmmaMat,
        operand: Staged,
        origin: Local,
        k: Operand,
    ) -> Vec<Local> {
        let (dtype, shape) = (self.wmma_dtype(), Self::shape());
        // The tile is `m x k` (A) or `k x n` (B): `X`-major puts its rows (A) or columns (B) contiguous.
        let major = match (mat, operand.major) {
            (WmmaMat::A, StagedMajor::X) | (WmmaMat::B, StagedMajor::K) => FragmentMajor::Row,
            (WmmaMat::A, StagedMajor::K) | (WmmaMat::B, StagedMajor::X) => FragmentMajor::Col,
        };
        let count = match mat {
            WmmaMat::A => self.rows,
            WmmaMat::B => self.cols,
        };
        (0..count as usize)
            .map(|f| {
                let x = bin(
                    e,
                    Ty::Usize,
                    BinOp::Add,
                    copy(origin),
                    cu(f * FRAGMENT_EDGE),
                );
                let base = match operand.major {
                    StagedMajor::X => {
                        let by_x = bin(e, Ty::Usize, BinOp::Mul, copy(x), cu(operand.stride));
                        bin(e, Ty::Usize, BinOp::Add, copy(by_x), k.clone())
                    }
                    StagedMajor::K => {
                        let by_k = bin(e, Ty::Usize, BinOp::Mul, k.clone(), cu(operand.stride));
                        bin(e, Ty::Usize, BinOp::Add, copy(by_k), copy(x))
                    }
                };
                let fragment = e.fresh(
                    Ty::Fragment(FragmentTy::matrix_major(mat, dtype, shape, major)),
                    false,
                );
                e.push(Statement::WmmaLoadLds {
                    role: FragmentRole::Matrix(mat),
                    dtype,
                    shape,
                    array: operand.array,
                    base: copy(base),
                    major,
                    stride: operand.stride as u32,
                    dst: fragment,
                });
                fragment
            })
            .collect()
    }

    /// One multiply per accumulator fragment: `acc[i][j] += a_frags[i] * b_frags[j]`.
    fn multiply(self, e: &mut Emit, acc: &mut Acc, a_frags: &[Local], b_frags: &[Local]) {
        let (dtype, shape) = (self.wmma_dtype(), Self::shape());
        for (i, &a_frag) in a_frags.iter().enumerate() {
            for (j, &b_frag) in b_frags.iter().enumerate() {
                let cell = acc.cells[i * b_frags.len() + j];
                e.push(Statement::WmmaMma {
                    dtype,
                    shape,
                    a: a_frag,
                    b: b_frag,
                    c: cell,
                    dst: cell,
                });
            }
        }
    }
}

/// The A fragments of one step a wave holds in registers across a loop ([`MicroTile::hold_a`]), one per fragment
/// row of its tile.
pub(crate) struct HeldA {
    fragments: Vec<Local>,
}

impl MicroTile {
    /// Load the wave's A fragments of staged depth `k` from `a` to hold them: the operand of every later
    /// [`Self::step_held`] at the same depth, however often `a`'s workgroup memory is reused in between. Only a
    /// fragment tile holds operands in registers.
    pub(crate) fn hold_a(self, e: &mut Emit, g: &Geometry, a: Staged, k: usize) -> HeldA {
        let Self::Fragment(tile) = self else {
            unreachable!("only a fragment tile holds its operands");
        };
        let wave = g.wave.as_ref().expect("a fragment geometry has a wave");
        HeldA {
            fragments: tile.load(e, WmmaMat::A, a, wave.row_origin, cu(k)),
        }
    }

    /// [`Self::step`] against held A fragments: `acc += held * B[k..k + 16, :]`, only `b` read from workgroup
    /// memory.
    pub(crate) fn step_held(
        self,
        e: &mut Emit,
        g: &Geometry,
        acc: &mut Acc,
        held: &HeldA,
        b: Staged,
        k: usize,
    ) {
        let Self::Fragment(tile) = self else {
            unreachable!("only a fragment tile holds its operands");
        };
        let wave = g.wave.as_ref().expect("a fragment geometry has a wave");
        let b_frags = tile.load(e, WmmaMat::B, b, wave.col_origin, cu(k));
        tile.multiply(e, acc, &held.fragments, &b_frags);
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

    /// Write every element `(row, col)` to `into` as element `(k = col, x = row)`: word `col * stride + row` of a
    /// `K`-major tile, `row * stride + col` of an `X`-major one. It is the `A` operand of a product whose depth is
    /// this tile's columns (the `P` of attention's `P * V`).
    ///
    /// `stored` turns each element into the value that is written (the narrowing of the operand it becomes).
    pub(crate) fn stage(
        &self,
        e: &mut Emit,
        into: Staged,
        mut stored: impl FnMut(&mut Emit, Local) -> Local,
    ) {
        for &(coord, cell) in &self.cells {
            let (outer, inner) = match into.major {
                StagedMajor::K => (coord.col, coord.row),
                StagedMajor::X => (coord.row, coord.col),
            };
            let by_outer = bin(e, Ty::Usize, BinOp::Mul, copy(outer), cu(into.stride));
            let idx = bin(e, Ty::Usize, BinOp::Add, copy(by_outer), copy(inner));
            let value = stored(e, cell);
            e.write_lds(into.array, copy(idx), copy(value));
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
