//! The [`super::Schedule::RegisterTiled`] loop nest over a weight whose rows are the output columns (`[N, K]`,
//! `K` contiguous): one body for every operand load. A packed front end supplies only the read of a run of
//! decoded weight values ([`tiled_body`]'s `weight_run`).
//!
//! A workgroup of `lanes x lanes` lanes owns a `(lanes * rows) x (lanes * MICRO_COLS)` output tile, and each lane
//! accumulates a `rows x MICRO_COLS` register tile of it. The registers are the point: a lane that owns one output
//! decodes one weight element and reads two LDS values per FMA, so the decode and the LDS traffic bound the
//! kernel; a lane that owns a register tile decodes each staged weight element once for all the tile's rows and
//! reads `rows + MICRO_COLS` LDS values per `rows * MICRO_COLS` FMAs. `K` is staged
//! `BLOCK_K` at a time, as runs of consecutive `K` values: a lane loads `run` activation values and decodes `run`
//! weight values per slot, so a weight run's address chain, word loads and shared scale are emitted once.

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, LocalDecl, Operand, Place,
    Rvalue, Statement, Terminator, Ty, WorkgroupLocalDecl,
};

use super::{LaneRows, RunLength, TileSize};
use crate::KernelGenError;
use crate::emit::{Emit, bin, copy, cu, emit_read_meta};
use crate::helpers::{Alloc, elem, guard};

/// Output columns along one lane's register tile.
const MICRO_COLS: usize = 4;

/// `K` elements staged in LDS per barrier round, and run in unrolled outer-product steps.
const BLOCK_K: usize = 16;

/// The CFG [`tiled_body`] builds: the `GroupX`, `GroupY` and `LocalX` reads, the setup block, the K loop (guard,
/// staged load, dot, latch) and then the store chain.
const GROUP_Y: u32 = 1;
const LANE: u32 = 2;
const SETUP: u32 = 3;
const LOOP_GUARD: u32 = 4;
const STAGE: u32 = 5;
const DOT: u32 = 6;
const LATCH: u32 = 7;
/// The first block of the store chain: one guard block and one store block per register-tile element.
const STORES: u32 = 8;

/// The rows of the output tile one workgroup of `lanes x lanes` lanes computes.
pub(super) const fn tile_rows(lanes: usize, rows: usize) -> usize {
    lanes * rows
}

/// The workgroup-local bytes one workgroup of `lanes x lanes` lanes of `rows` rows each stages its two tiles in.
pub(super) const fn lds_bytes(lanes: usize, rows: usize) -> usize {
    BLOCK_K * (tile_rows(lanes, rows) + 1 + tile_cols(lanes) + 1) * std::mem::size_of::<f32>()
}

/// The columns of the output tile one workgroup of `lanes x lanes` lanes computes.
pub(super) const fn tile_cols(lanes: usize) -> usize {
    lanes * MICRO_COLS
}

/// A [`super::Schedule::RegisterTiled`]'s launch shape, checked once before emission: both staged tiles must split
/// into whole runs over the lanes, so every lane stages the same number of runs of each.
#[derive(Clone, Copy)]
pub(crate) struct TiledLaunch {
    lanes: usize,
    rows: usize,
    run: usize,
}

impl TiledLaunch {
    pub(crate) fn new(
        lanes: TileSize,
        rows: LaneRows,
        run: RunLength,
    ) -> Result<Self, KernelGenError> {
        let (lanes, rows, run) = (
            lanes.get() as usize,
            rows.get() as usize,
            run.get() as usize,
        );
        for (what, edge) in [
            ("rows", tile_rows(lanes, rows)),
            ("columns", tile_cols(lanes)),
        ] {
            let runs = edge * (BLOCK_K / run);
            if !runs.is_multiple_of(lanes * lanes) {
                return Err(KernelGenError::NotDivisible {
                    generator: "contraction_register_tiled",
                    dim: format!("staged {what} runs"),
                    value: runs,
                    divisor: lanes * lanes,
                });
            }
        }
        Ok(Self { lanes, rows, run })
    }

    /// Runs per row of a staged tile: a row's `BLOCK_K` values are `BLOCK_K / run` runs.
    fn runs_per_row(self) -> usize {
        BLOCK_K / self.run
    }

    /// Runs of a staged tile of `edge` rows each lane loads per round.
    fn slots(self, edge: usize) -> usize {
        edge * self.runs_per_row() / (self.lanes * self.lanes)
    }
}

/// The body's buffers: the full parameter list (`unit`, the activation, the weight sources, `metadata`, `out`) and
/// where the three the loop nest itself reads sit.
pub(crate) struct TiledBody<'a> {
    pub(crate) name: &'a str,
    pub(crate) params: Vec<LocalDecl>,
    pub(crate) activation: Local,
    pub(crate) metadata: Local,
    pub(crate) out: Local,
}

/// One staged run of one operand's tile: where its `run` values land in LDS and the row they come from.
struct StagedRun {
    /// The run's first `K` offset within the round.
    k_offset: Local,
    /// The LDS slot of the run's first value; value `j` is `j` rows (`stride` slots each) further.
    lds_idx: Local,
    /// The activation row's first element (A) or the weight row (B).
    source_row: Local,
}

/// The [`super::Schedule::RegisterTiled`] loop nest, shared by every operand load; `weight_run(e, weight_row, k0,
/// count, k, weight_rows)` emits the `count` logical f32 weight values `k0 .. k0 + count` of row `weight_row` of a
/// weight with `weight_rows` rows of `k` values, `k0` a multiple of `count`, and is only ever called with the
/// whole run inside `0..k` (`k` is a multiple of the launch's `run`, so a run is whole or wholly past the end).
///
/// `metadata` is `[K, out_per_block, m_per_block, x_groups, tile_offset]`; `activation` is `[rows, K]` and `out` is
/// `[rows, out_per_block]` with `rows = blocks * m_per_block` (block-diagonal: block `b`'s rows read weight rows
/// `b * out_per_block ..`). A flat tile index `g = tile_offset + GroupY * x_groups + GroupX` (`tile_offset` is the first tile of the range
/// this dispatch covers: a contraction over the device's watchdog budget is several dispatches over disjoint tile
/// ranges, 0 when unsplit) is laid out block-major, so one
/// workgroup's staged weight tile is always one block's, and every edge read is clamped in range, never branched
/// on inside the barrier'd loop: a row or column past the block re-reads a real one (its output is never stored,
/// and an output depends on its own row and column alone), and a `K` tail re-reads the last run, which the weight
/// tile zeroes. A workgroup past the last tile of a folded grid reads the last real row and never stores.
///
/// The `K` tail is zeroed on the weight tile only, so an activation `Inf`/`NaN` in the last real run reaches the
/// tail slots as `Inf * 0 = NaN`; that output was already non-finite for any nonzero weight there. Only a planar
/// format has a `K` tail (a block format's `K` is a multiple of `BLOCK_K`), so only the planar rows exercise the
/// mask.
pub(crate) fn tiled_body(
    spec: TiledBody<'_>,
    launch: TiledLaunch,
    mut weight_run: impl FnMut(&mut Emit, Local, Local, usize, Local, Local) -> Vec<Local>,
) -> Body {
    let TiledBody {
        name,
        params,
        activation,
        metadata,
        out,
    } = spec;
    let (lanes, micro_rows, run) = (launch.lanes, launch.rows, launch.run);
    let (edge_m, edge_n) = (tile_rows(lanes, micro_rows), tile_cols(lanes));
    // One LDS row of a staged tile is padded by one word, so the cooperative stores (adjacent lanes adjacent in
    // the tile's rows or columns) and the dot reads (adjacent lanes adjacent in a register tile) both spread
    // over the banks.
    let (stride_a, stride_b) = (edge_m + 1, edge_n + 1);
    let lane_count = lanes * lanes;
    let n_inputs = (params.len() - 1) as u32;
    let mut al = Alloc::new(params);
    let gx = al.add(Ty::Usize, false);
    let gy = al.add(Ty::Usize, false);
    let l = al.add(Ty::Usize, false);
    let kt = al.add(Ty::Usize, true);
    let acc: Vec<Local> = (0..micro_rows * MICRO_COLS)
        .map(|_| al.add(Ty::F32, true))
        .collect();

    let thread_index = |destination: Local, dim: IndexAxis, target: u32| BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(destination),
            dim,
            target: BlockId { index: target },
        },
    };
    let goto = |target: u32| Terminator::Goto {
        target: BlockId { index: target },
    };
    let barrier = |target: u32| Terminator::Barrier {
        target: BlockId { index: target },
    };

    // --- setup: tile coordinates, the loop-invariant parts of every staged run, the accumulators ---
    let mut setup = Emit::new(&mut al);
    let k = emit_read_meta(&mut setup, metadata, 0);
    let out_per_block = emit_read_meta(&mut setup, metadata, 1);
    let m_per_block = emit_read_meta(&mut setup, metadata, 2);
    let x_groups = emit_read_meta(&mut setup, metadata, 3);
    let g = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(gy), copy(x_groups));
    let g = bin(&mut setup, Ty::Usize, BinOp::Add, copy(g), copy(gx));
    let tile_offset = emit_read_meta(&mut setup, metadata, 4);
    let g = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Add,
        copy(g),
        copy(tile_offset),
    );

    // `rows` and `blocks` come from `out`'s own length, never a metadata word: `out` is exactly
    // `[rows, out_per_block]`.
    let out_len = setup.let_(Ty::Usize, Rvalue::Len(Place::local(out)));
    let rows = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(out_len),
        copy(out_per_block),
    );
    let blocks = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(rows),
        copy(m_per_block),
    );
    let weight_rows = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(blocks),
        copy(out_per_block),
    );
    let max_weight_row = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(weight_rows), cu(1));
    let max_row = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(rows), cu(1));
    // The last run start: `k` is a multiple of `run`, so every run is whole or wholly past the end, and a run past
    // the end re-reads the last whole one (its values are masked to zero).
    let last_run = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(k), cu(run));

    let ceil_div = |e: &mut Emit, value: Local, divisor: usize| {
        let padded = bin(e, Ty::Usize, BinOp::Add, copy(value), cu(divisor - 1));
        bin(e, Ty::Usize, BinOp::Div, copy(padded), cu(divisor))
    };
    let tiles_col = ceil_div(&mut setup, out_per_block, edge_n);
    let tiles_row = ceil_div(&mut setup, m_per_block, edge_m);
    let kt_count = ceil_div(&mut setup, k, BLOCK_K);
    let tiles_per_block = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(tiles_row),
        copy(tiles_col),
    );
    let block = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(g),
        copy(tiles_per_block),
    );
    let tile_in_block = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Rem,
        copy(g),
        copy(tiles_per_block),
    );
    let tile_row = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(tile_in_block),
        copy(tiles_col),
    );
    let tile_col = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Rem,
        copy(tile_in_block),
        copy(tiles_col),
    );
    let row_base = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(tile_row),
        cu(edge_m),
    );
    let col_base = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(tile_col),
        cu(edge_n),
    );
    let block_row0 = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(block),
        copy(m_per_block),
    );
    let block_weight_row0 = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(block),
        copy(out_per_block),
    );

    let lane_row = bin(&mut setup, Ty::Usize, BinOp::Div, copy(l), cu(lanes));
    let lane_col = bin(&mut setup, Ty::Usize, BinOp::Rem, copy(l), cu(lanes));
    // This lane's register tile sits at `lane_row + lanes * i` of the tile's rows and `lane_col + lanes * j` of its
    // columns (interleaved, so adjacent lanes read adjacent LDS words).
    let lane_rows: Vec<Local> = (0..micro_rows)
        .map(|i| {
            bin(
                &mut setup,
                Ty::Usize,
                BinOp::Add,
                copy(lane_row),
                cu(lanes * i),
            )
        })
        .collect();
    let lane_cols: Vec<Local> = (0..MICRO_COLS)
        .map(|j| {
            bin(
                &mut setup,
                Ty::Usize,
                BinOp::Add,
                copy(lane_col),
                cu(lanes * j),
            )
        })
        .collect();

    // Staged run `q` of a tile of `edge` rows: run `r = q * lane_count + l` is row `r / runs_per_row`, `K` offset
    // `(r % runs_per_row) * run` (adjacent lanes adjacent in `K`: the coalesced direction of both operands).
    // `base` is the tile's first row in its block; `source_row0` and `source_stride` turn the row into the buffer row
    // the run reads, clamped to the last real one. A row past the block (or the buffer) re-reads a real row: its
    // output is never stored, and each output depends on its own row and column alone, so no edge needs a mask.
    let stage_runs = |setup: &mut Emit,
                      edge: usize,
                      stride: usize,
                      base: Local,
                      source_row0: Local,
                      max_source_row: Local,
                      source_stride: Option<Local>|
     -> Vec<StagedRun> {
        let runs_per_row = launch.runs_per_row();
        (0..launch.slots(edge))
            .map(|q| {
                let r = bin(setup, Ty::Usize, BinOp::Add, copy(l), cu(q * lane_count));
                let k_chunk = bin(setup, Ty::Usize, BinOp::Rem, copy(r), cu(runs_per_row));
                let k_offset = bin(setup, Ty::Usize, BinOp::Mul, copy(k_chunk), cu(run));
                let tile_idx = bin(setup, Ty::Usize, BinOp::Div, copy(r), cu(runs_per_row));
                let lds_row = bin(setup, Ty::Usize, BinOp::Mul, copy(k_offset), cu(stride));
                let lds_idx = bin(setup, Ty::Usize, BinOp::Add, copy(lds_row), copy(tile_idx));

                let in_block = bin(setup, Ty::Usize, BinOp::Add, copy(base), copy(tile_idx));
                let source_row = bin(
                    setup,
                    Ty::Usize,
                    BinOp::Add,
                    copy(source_row0),
                    copy(in_block),
                );
                let source_row = bin(
                    setup,
                    Ty::Usize,
                    BinOp::Min,
                    copy(source_row),
                    copy(max_source_row),
                );
                let source_row = match source_stride {
                    Some(stride) => {
                        bin(setup, Ty::Usize, BinOp::Mul, copy(source_row), copy(stride))
                    }
                    None => source_row,
                };
                StagedRun {
                    k_offset,
                    lds_idx,
                    source_row,
                }
            })
            .collect()
    };
    let a_runs = stage_runs(
        &mut setup,
        edge_m,
        stride_a,
        row_base,
        block_row0,
        max_row,
        Some(k),
    );
    let b_runs = stage_runs(
        &mut setup,
        edge_n,
        stride_b,
        col_base,
        block_weight_row0,
        max_weight_row,
        None,
    );

    let mut setup_stmts = setup.take();
    let zero = Operand::Const(Constant::F32(0.0));
    for &acc in &acc {
        setup_stmts.push(Statement::Assign(
            Place::local(acc),
            Rvalue::Use(zero.clone()),
        ));
    }
    setup_stmts.push(Statement::Assign(Place::local(kt), Rvalue::Use(cu(0))));
    let setup_block = BasicBlock {
        statements: setup_stmts,
        terminator: goto(LOOP_GUARD),
    };

    let mut guard_e = Emit::new(&mut al);
    let more = bin(&mut guard_e, Ty::Bool, BinOp::Lt, copy(kt), copy(kt_count));
    let loop_guard = BasicBlock {
        statements: guard_e.take(),
        terminator: guard(more, STORES, STAGE),
    };

    // --- stage: every lane loads its activation runs and decodes its weight runs into LDS ---
    let mut load = Emit::new(&mut al);
    let kt_base = bin(&mut load, Ty::Usize, BinOp::Mul, copy(kt), cu(BLOCK_K));
    // A run's start, clamped to the last whole run: a run past the end of `K` re-reads it. The weight tile is
    // zeroed there (`k_mask`), so the activation tile needs no mask of its own.
    let run_start = |load: &mut Emit, k_offset: Local| {
        let start = bin(load, Ty::Usize, BinOp::Add, copy(kt_base), copy(k_offset));
        let clamped = bin(load, Ty::Usize, BinOp::Min, copy(start), copy(last_run));
        (start, clamped)
    };
    for staged in &a_runs {
        let (_, start) = run_start(&mut load, staged.k_offset);
        let first = bin(
            &mut load,
            Ty::Usize,
            BinOp::Add,
            copy(staged.source_row),
            copy(start),
        );
        for j in 0..run {
            let idx = bin(&mut load, Ty::Usize, BinOp::Add, copy(first), cu(j));
            let value = load.let_(Ty::F32, Rvalue::Use(Operand::Copy(elem(activation, idx))));
            let slot = bin(
                &mut load,
                Ty::Usize,
                BinOp::Add,
                copy(staged.lds_idx),
                cu(j * stride_a),
            );
            load.write_lds(0, copy(slot), copy(value));
        }
    }
    for staged in &b_runs {
        let (unclamped, start) = run_start(&mut load, staged.k_offset);
        let in_range = bin(&mut load, Ty::Bool, BinOp::Lt, copy(unclamped), copy(k));
        let k_mask = load.let_(
            Ty::F32,
            Rvalue::Cast {
                to: Ty::F32,
                operand: copy(in_range),
            },
        );
        let values = weight_run(&mut load, staged.source_row, start, run, k, weight_rows);
        for (j, &decoded) in values.iter().enumerate() {
            let value = bin(&mut load, Ty::F32, BinOp::Mul, copy(decoded), copy(k_mask));
            let slot = bin(
                &mut load,
                Ty::Usize,
                BinOp::Add,
                copy(staged.lds_idx),
                cu(j * stride_b),
            );
            load.write_lds(1, copy(slot), copy(value));
        }
    }
    let stage = BasicBlock {
        statements: load.take(),
        terminator: barrier(DOT),
    };

    // --- dot: `BLOCK_K` unrolled outer-product steps over the two staged tiles ---
    let mut dot = Emit::new(&mut al);
    for step in 0..BLOCK_K {
        let a_vals: Vec<Local> = lane_rows
            .iter()
            .map(|&lane_row| {
                let idx = bin(
                    &mut dot,
                    Ty::Usize,
                    BinOp::Add,
                    cu(step * stride_a),
                    copy(lane_row),
                );
                dot.read_lds(Ty::F32, 0, copy(idx))
            })
            .collect();
        let b_vals: Vec<Local> = lane_cols
            .iter()
            .map(|&lane_col| {
                let idx = bin(
                    &mut dot,
                    Ty::Usize,
                    BinOp::Add,
                    cu(step * stride_b),
                    copy(lane_col),
                );
                dot.read_lds(Ty::F32, 1, copy(idx))
            })
            .collect();
        for (i, &a_val) in a_vals.iter().enumerate() {
            for (j, &b_val) in b_vals.iter().enumerate() {
                let cell = acc[i * MICRO_COLS + j];
                let product = bin(&mut dot, Ty::F32, BinOp::Mul, copy(a_val), copy(b_val));
                let next = bin(&mut dot, Ty::F32, BinOp::Add, copy(cell), copy(product));
                dot.assign(Place::local(cell), Rvalue::Use(copy(next)));
            }
        }
    }
    let dot = BasicBlock {
        statements: dot.take(),
        terminator: goto(LATCH),
    };

    let mut latch = Emit::new(&mut al);
    let kt_next = bin(&mut latch, Ty::Usize, BinOp::Add, copy(kt), cu(1));
    latch.assign(Place::local(kt), Rvalue::Use(copy(kt_next)));
    let latch = BasicBlock {
        statements: latch.take(),
        terminator: barrier(LOOP_GUARD),
    };

    // --- stores: one guarded store per register-tile element; a row or column past the block, or a block past the
    // last (a folded grid's padding workgroups), stores nothing ---
    let mut blocks_out = vec![
        thread_index(gx, IndexAxis::GroupX, GROUP_Y),
        thread_index(gy, IndexAxis::GroupY, LANE),
        thread_index(l, IndexAxis::LocalX, SETUP),
        setup_block,
        loop_guard,
        stage,
        dot,
        latch,
    ];
    debug_assert_eq!(blocks_out.len() as u32, STORES);
    for i in 0..micro_rows {
        for j in 0..MICRO_COLS {
            let n = (i * MICRO_COLS + j) as u32;
            let (store, next) = (STORES + 2 * n + 1, STORES + 2 * n + 2);

            let mut cond = Emit::new(&mut al);
            let row_in_block = bin(
                &mut cond,
                Ty::Usize,
                BinOp::Add,
                copy(row_base),
                copy(lane_rows[i]),
            );
            let col = bin(
                &mut cond,
                Ty::Usize,
                BinOp::Add,
                copy(col_base),
                copy(lane_cols[j]),
            );
            let row = bin(
                &mut cond,
                Ty::Usize,
                BinOp::Add,
                copy(block_row0),
                copy(row_in_block),
            );
            let row_ok = bin(
                &mut cond,
                Ty::Bool,
                BinOp::Lt,
                copy(row_in_block),
                copy(m_per_block),
            );
            let col_ok = bin(
                &mut cond,
                Ty::Bool,
                BinOp::Lt,
                copy(col),
                copy(out_per_block),
            );
            let in_range = bin(&mut cond, Ty::Bool, BinOp::Lt, copy(row), copy(rows));
            let tile_ok = bin(
                &mut cond,
                Ty::Bool,
                BinOp::BitAnd,
                copy(row_ok),
                copy(col_ok),
            );
            let store_ok = bin(
                &mut cond,
                Ty::Bool,
                BinOp::BitAnd,
                copy(tile_ok),
                copy(in_range),
            );
            blocks_out.push(BasicBlock {
                statements: cond.take(),
                terminator: guard(store_ok, next, store),
            });

            let mut write = Emit::new(&mut al);
            let row_off = bin(
                &mut write,
                Ty::Usize,
                BinOp::Mul,
                copy(row),
                copy(out_per_block),
            );
            let out_idx = bin(&mut write, Ty::Usize, BinOp::Add, copy(row_off), copy(col));
            write.assign(
                elem(out, out_idx),
                Rvalue::Use(copy(acc[i * MICRO_COLS + j])),
            );
            blocks_out.push(BasicBlock {
                statements: write.take(),
                terminator: goto(next),
            });
        }
    }
    blocks_out.push(BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    });

    let mut body = Body::new(name, n_inputs, al.locals, blocks_out);
    body.workgroup_size = [lane_count as u32, 1, 1];
    body.workgroup_locals = [stride_a, stride_b]
        .into_iter()
        .map(|stride| WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: (BLOCK_K * stride) as u32,
        })
        .collect();
    body
}
