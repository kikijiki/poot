//! `packed_kernel`: the public entry point. One generated [`Body`] per `(format, op)`
//! ([`PackedKernelSpec`]) serves every shape: `K` and the launch extent ride in the `metadata`
//! parameter (a `&[usize]` buffer the planner fills per call), never baked into the body, so the
//! codegen cache keys on the spec alone (dquant.md 3.2, 5).

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, IndexAxis, Local, Operand, Place, Rvalue, Statement,
    Terminator, Ty,
};
use poot_quant::format::{Storage, WeightFormat};

use super::decode_plan::DecodePlan;
use super::planar::PlanarPlan;
use crate::KernelGenError;
use crate::contraction::{GemvLaunch, Schedule, TileSize, gemv_body};
use crate::emit::{Emit, bin, copy, cu, emit_read_meta};
use crate::helpers::{Alloc, elem, guard, ld, local, slice_dtype, slice_f32};

/// What one generated packed kernel computes (dquant.md 3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedKernelSpec {
    pub format: WeightFormat,
    pub op: PackedKernelOp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackedKernelOp {
    /// `out[o, k] = decode(o, k)`: a standalone `PackedDequant`. The tier-1 device receipt.
    Materialize,
    /// `out[r, :] = decode(ids[r], :)`: `Gather(axis 0)` over `PackedDequant`.
    RowGather,
    /// `PackedContraction` and the ADR-0109 indexed and grouped forms.
    Contraction { rows: RowSelect, schedule: Schedule },
}

/// Which weight rows an output row reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowSelect {
    Dense,
    Indexed,
    Grouped,
}

/// `metadata[0]`: every generated body reads `K` (and further shape facts, by op) from this
/// parameter rather than baking a shape into the body (see the module docs).
const META_K: usize = 0;

pub fn packed_kernel(name: &str, spec: PackedKernelSpec) -> Result<Body, KernelGenError> {
    match spec.op {
        PackedKernelOp::Materialize => materialize(name, spec.format),
        PackedKernelOp::RowGather => row_gather(name, spec.format),
        PackedKernelOp::Contraction { rows, schedule } => {
            if rows != RowSelect::Dense {
                return Err(KernelGenError::UnsupportedDescriptor {
                    format: spec.format,
                    reason: "RowSelect::Indexed/Grouped are not yet generated (542a covers Dense)",
                });
            }
            match schedule {
                Schedule::Gemv {
                    width,
                    cols,
                    unroll,
                } => contraction_gemv(name, spec.format, GemvLaunch::new(width, cols, unroll)?),
                Schedule::Tiled { tile } => contraction_tiled(name, spec.format, tile),
            }
        }
    }
}

/// `block = row*(k/values) + k_index/values`, `i_in_block = k_index % values`: the block-32 address
/// of logical `(row, k_index)` for a weight whose row holds `k` (total) logical values in blocks of
/// `values` (`DecodePlan::values`). Shared by every op ([`materialize`], [`row_gather`],
/// [`contraction_gemv`], `contraction_tiled`) so the row-major block layout has one KIR
/// implementation, matching [`poot_quant::PackedWeight::source_shape`]'s host-side accounting.
fn emit_block_address(
    e: &mut Emit,
    values: usize,
    row: Local,
    k_index: Local,
    k: Local,
) -> (Local, Local) {
    let row_blocks = bin(e, Ty::Usize, BinOp::Div, copy(k), cu(values));
    let block_in_row = bin(e, Ty::Usize, BinOp::Div, copy(k_index), cu(values));
    let row_base = bin(e, Ty::Usize, BinOp::Mul, copy(row), copy(row_blocks));
    let block = bin(e, Ty::Usize, BinOp::Add, copy(row_base), copy(block_in_row));
    let i_in_block = bin(e, Ty::Usize, BinOp::Rem, copy(k_index), cu(values));
    (block, i_in_block)
}

/// `out[o, k] = decode(o, k)` for every element of a `[out_rows, K]` weight: one thread per output
/// element, params `(words, metadata, out)`. `metadata[META_K]` is `K`; `out.len()` gives the launch
/// extent and (with `K`) `out_rows`, so nothing about the shape is baked into the body.
fn materialize(name: &str, format: WeightFormat) -> Result<Body, KernelGenError> {
    let descriptor = format.descriptor();
    if let Storage::Planar(_) = descriptor.storage {
        return materialize_planar(name, format);
    }
    let plan = DecodePlan::of(&descriptor)?;
    let values = plan.values();

    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_f32(true), true),
    ]);
    let (words, metadata, out) = (local(1), local(2), local(3));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(out))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(i), copy(len)),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };

    let mut emit = Emit::new(&mut al);
    let k = emit_read_meta(&mut emit, metadata, META_K);
    let row = emit.let_(Ty::Usize, Rvalue::BinaryOp(BinOp::Div, copy(i), copy(k)));
    let kk = emit.let_(Ty::Usize, Rvalue::BinaryOp(BinOp::Rem, copy(i), copy(k)));
    let (block, i_in_block) = emit_block_address(&mut emit, values, row, kk, k);

    let factors = plan.emit_factors(&mut emit, words, block, i_in_block);
    let value = plan.emit_value(&mut emit, words, block, i_in_block, &factors);
    emit.assign(elem(out, i), Rvalue::Use(copy(value)));

    let bb2 = BasicBlock {
        statements: emit.take(),
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Ok(Body::new(name, 3, al.locals, vec![bb0, bb1, bb2, bb3]))
}

/// [`materialize`]'s planar front end (card 542c, dquant.md 3.3): the same `out[o, k] = decode(o,
/// k)` shape, but `words` becomes one `&[u32]` parameter per `PlanarPlan::source_count()` (in
/// `PackedWeight::sources()` order), and the address comes from `PlanarPlan::emit_factors`/
/// `emit_value` directly off `(o, k)` rather than a block/in-block-index pair. `out_total` (a
/// planar operand's `out`-axis stored extent needs it; a block format never does) is `len(out) /
/// K`, since `out`'s shape is `[out_total, K]` flattened, the same fact `contraction_tiled`
/// already derives from its own output buffer's length.
fn materialize_planar(name: &str, format: WeightFormat) -> Result<Body, KernelGenError> {
    let descriptor = format.descriptor();
    let plan = PlanarPlan::of(&descriptor)?;
    let n = plan.source_count();

    let mut params = vec![ld(Ty::Unit, false)];
    for _ in 0..n {
        params.push(ld(slice_dtype(Ty::U32, false), false));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_f32(true), true));
    let mut al = Alloc::new(params);
    let sources: Vec<Local> = (0..n as u32).map(|idx| local(1 + idx)).collect();
    let metadata = local(1 + n as u32);
    let out = local(2 + n as u32);
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(out))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(i), copy(len)),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };

    let mut emit = Emit::new(&mut al);
    let k = emit_read_meta(&mut emit, metadata, META_K);
    let row = emit.let_(Ty::Usize, Rvalue::BinaryOp(BinOp::Div, copy(i), copy(k)));
    let kk = emit.let_(Ty::Usize, Rvalue::BinaryOp(BinOp::Rem, copy(i), copy(k)));
    let out_total = bin(&mut emit, Ty::Usize, BinOp::Div, copy(len), copy(k));

    let factors = plan.emit_factors(&mut emit, &sources, row, kk, k, out_total);
    let value = plan.emit_value(&mut emit, &sources, row, kk, k, out_total, &factors);
    emit.assign(elem(out, i), Rvalue::Use(copy(value)));

    let bb2 = BasicBlock {
        statements: emit.take(),
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Ok(Body::new(
        name,
        2 + n as u32,
        al.locals,
        vec![bb0, bb1, bb2, bb3],
    ))
}

/// `out[r, :] = decode(ids[r], :)` for a `[R, K]` output, `ids` an `R`-length row index (`f32`, the
/// executors' index-param convention: token ids are bound as f32 and cast in-kernel):
/// [`PackedKernelOp::RowGather`], the axis-0 `Gather` fused directly onto a `PackedDequant` producer
/// (GGUF token embeddings). Params `(words, ids, metadata, out)`, one thread per output
/// element; identical to [`materialize`] except the row comes from `ids[r]` instead of the flat index.
fn row_gather(name: &str, format: WeightFormat) -> Result<Body, KernelGenError> {
    let descriptor = format.descriptor();
    if let Storage::Planar(_) = descriptor.storage {
        return Err(KernelGenError::UnsupportedDescriptor {
            format,
            reason: "planar RowGather (542c's table admits planar Materialize/Contraction only; \
                     no production consumer gathers planar-format rows today)",
        });
    }
    let plan = DecodePlan::of(&descriptor)?;
    let values = plan.values();

    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_f32(false), false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_f32(true), true),
    ]);
    let (words, ids, metadata, out) = (local(1), local(2), local(3), local(4));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let cmp = al.add(Ty::Bool, false);

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(out))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(i), copy(len)),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };

    let mut emit = Emit::new(&mut al);
    let k = emit_read_meta(&mut emit, metadata, META_K);
    let r = emit.let_(Ty::Usize, Rvalue::BinaryOp(BinOp::Div, copy(i), copy(k)));
    let kk = emit.let_(Ty::Usize, Rvalue::BinaryOp(BinOp::Rem, copy(i), copy(k)));
    let row_id = emit.let_(Ty::F32, Rvalue::Use(Operand::Copy(elem(ids, r))));
    let row = emit.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(row_id),
        },
    );
    let (block, i_in_block) = emit_block_address(&mut emit, values, row, kk, k);

    let factors = plan.emit_factors(&mut emit, words, block, i_in_block);
    let value = plan.emit_value(&mut emit, words, block, i_in_block, &factors);
    emit.assign(elem(out, i), Rvalue::Use(copy(value)));

    let bb2 = BasicBlock {
        statements: emit.take(),
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Ok(Body::new(name, 4, al.locals, vec![bb0, bb1, bb2, bb3]))
}

/// `out[col_flat] = sum over k of activation[b, k] * decode(col_flat, k)`, `M == 1`:
/// [`Schedule::Gemv`] over the block-storage front end. `col_flat` ranges over `blocks *
/// out_per_block` (the flattened `[blocks, out_per_block]` output, which is also the weight row
/// index; `blocks == 1` is the ordinary dense GEMV) and `b = col_flat / out_per_block`. Params
/// `(activation, words, metadata, out)` - activation first, matching `OpKind::PackedContraction`'s
/// own operand order (card 542a production wiring, D9). `metadata` is `[K, out_per_block]`;
/// `activation` is `[blocks, K]` (one row per block, `M == 1`).
fn contraction_gemv(
    name: &str,
    format: WeightFormat,
    launch: GemvLaunch,
) -> Result<Body, KernelGenError> {
    let descriptor = format.descriptor();
    if let Storage::Planar(_) = descriptor.storage {
        return contraction_gemv_planar(name, format, launch);
    }
    let plan = DecodePlan::of(&descriptor)?;
    let values = plan.values();
    let al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_f32(true), true),
    ]);
    let (activation, words, metadata, out) = (local(1), local(2), local(3), local(4));
    // A run of `count` elements starting at a multiple of `count` stays inside one block when
    // `count` divides the block, so the block is addressed once; when `count` also divides the
    // factor run, the run's factors are decoded once and shared by all its elements. Otherwise each
    // element addresses its own block and decodes its own factors.
    let factor_run = plan.factor_run();
    let decode_run = |e: &mut Emit, row: Local, k0: Local, count: usize, k: Local, _: Local| {
        let run_block = values
            .is_multiple_of(count)
            .then(|| emit_block_address(e, values, row, k0, k));
        let run_factors = run_block
            .filter(|_| factor_run.is_multiple_of(count))
            .map(|(block, first)| plan.emit_factors(e, words, block, first));
        (0..count)
            .map(|j| {
                let (block, i_in_block) = match run_block {
                    Some((block, first)) => {
                        (block, bin(e, Ty::Usize, BinOp::Add, copy(first), cu(j)))
                    }
                    None => {
                        let k_index = bin(e, Ty::Usize, BinOp::Add, copy(k0), cu(j));
                        emit_block_address(e, values, row, k_index, k)
                    }
                };
                let own_factors;
                let factors = match &run_factors {
                    Some(factors) => factors,
                    None => {
                        own_factors = plan.emit_factors(e, words, block, i_in_block);
                        &own_factors
                    }
                };
                plan.emit_value(e, words, block, i_in_block, factors)
            })
            .collect()
    };
    Ok(gemv_body(
        name,
        al,
        4,
        [activation, metadata, out],
        launch,
        decode_run,
    ))
}

/// [`contraction_gemv`]'s planar front end (card 542c): the same schedule, `words` becomes
/// `PlanarPlan::source_count()` parameters, the per-element decode reads `PlanarPlan` off
/// `(weight_row, k)`. `out_total` is `len(out)` (`M == 1` gives `rows == blocks`, so `out`'s
/// `[blocks, out_per_block]` flattens to exactly the weight's true row count).
fn contraction_gemv_planar(
    name: &str,
    format: WeightFormat,
    launch: GemvLaunch,
) -> Result<Body, KernelGenError> {
    let descriptor = format.descriptor();
    let plan = PlanarPlan::of(&descriptor)?;
    let n = plan.source_count();
    let mut params = vec![ld(Ty::Unit, false), ld(slice_f32(false), false)];
    for _ in 0..n {
        params.push(ld(slice_dtype(Ty::U32, false), false));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_f32(true), true));
    let al = Alloc::new(params);
    let activation = local(1);
    let sources: Vec<Local> = (0..n as u32).map(|idx| local(2 + idx)).collect();
    let metadata = local(2 + n as u32);
    let out = local(3 + n as u32);
    let decode_run =
        |e: &mut Emit, row: Local, k0: Local, count: usize, k: Local, out_total: Local| {
            (0..count)
                .map(|j| {
                    let k_index = bin(e, Ty::Usize, BinOp::Add, copy(k0), cu(j));
                    let factors = plan.emit_factors(e, &sources, row, k_index, k, out_total);
                    plan.emit_value(e, &sources, row, k_index, k, out_total, &factors)
                })
                .collect()
        };
    Ok(gemv_body(
        name,
        al,
        3 + n as u32,
        [activation, metadata, out],
        launch,
        decode_run,
    ))
}

/// `out[row, col] = sum over k of activation[row, k] * decode(weight_row(row, col), k)`, `M > 1`:
/// [`Schedule::Tiled`] (card 542b, dquant.md 3.2's `load_masked` seam generalized to the descriptor
/// decode). A square `tile x tile` workgroup cooperatively loads one activation tile and one decoded
/// weight tile into LDS per K sub-tile (each lane decodes exactly one weight element through
/// [`DecodePlan`], never a per-format unpack), barriers, does an unrolled `tile`-wide dot, barriers
/// again, and advances to the next K sub-tile. Every tile-edge read (`row_in_block >= m_per_block`,
/// `a_k >= K`, `col >= out_per_block`, `b_k >= K`) is masked to zero after a clamped, in-bounds read,
/// never branched on - the same RADV-safe shape `tiled_region_impl` (the hand-written GGUF
/// `TiledRegionDequant` family this schedule's production callers replaced, dquant.md D6/Q9, since
/// deleted by card 545b) uses for its own cooperative load. Performance (K-unroll, multi-row-per-lane
/// coarsening) is out of this card's scope; only one output element per lane, one K-tile per barrier
/// round. Params `(activation, words, metadata, out)` - activation first, matching
/// `OpKind::PackedContraction`'s own operand order (card 542a production wiring, D9); `metadata` is
/// `[K, out_per_block, m_per_block, x_groups]` (card 658: `x_groups` is the fold-aware
/// grid's X-dim tile count, read unconditionally below), `activation` is `[rows, K]` (`rows = blocks *
/// m_per_block`, block-diagonal).
///
/// **Card 658: tile rows never cross a block.** A flat `GroupX` tile index `g` is laid out block-major
/// (`blocks * ceil(m_per_block/tile) * ceil(out_per_block/tile)` tiles total,
/// [`Schedule::grid_threads`]'s `Tiled` arm), so `b = g / tiles_per_block` is fixed for every lane in
/// a workgroup, never derived from a per-lane row. The one shared `Bs` tile this schedule's cooperative
/// load stages is therefore always one block's weight, regardless of whether `m_per_block` is a
/// multiple of `tile`: a tile's row-local offset (`row_in_block`) is masked against `m_per_block`
/// directly (not against the flattened `rows`), so a padded tail row can never alias into the next
/// block's rows. `Schedule::Gemv`'s workgroup is exactly one row, so it never had this hazard either.
fn contraction_tiled(
    name: &str,
    format: WeightFormat,
    tile: TileSize,
) -> Result<Body, KernelGenError> {
    let tile = tile.get();
    let descriptor = format.descriptor();
    if let Storage::Planar(_) = descriptor.storage {
        return contraction_tiled_planar(name, format, tile);
    }
    let plan = DecodePlan::of(&descriptor)?;
    let values = plan.values();
    let ts = tile as usize;

    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_f32(true), true),
    ]);
    let (activation, words, metadata, out) = (local(1), local(2), local(3), local(4));
    let gx = al.add(Ty::Usize, false);
    let gy = al.add(Ty::Usize, false);
    let l = al.add(Ty::Usize, false);

    // Card 658 review F1/F2: `gx`/`gy` are always read, never just `GroupX` alone, so this body
    // survives a folded (2-D) dispatch grid above `caps.max_grid[0]` tiles - see `g`'s derivation in
    // `setup` below and `try_plan_contraction`'s doc. The GroupY read is appended as its own block
    // (index 10) rather than renumbering bb1.. so every other block index stays stable.
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gx),
            dim: IndexAxis::GroupX,
            target: BlockId { index: 10 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(l),
            dim: IndexAxis::LocalX,
            target: BlockId { index: 2 },
        },
    };

    let mut setup = Emit::new(&mut al);
    let k = emit_read_meta(&mut setup, metadata, META_K);
    let meta1 = setup.let_(Ty::Usize, Rvalue::Use(cu(1)));
    let out_per_block_u32 = setup.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(metadata, meta1))));
    let out_per_block = setup.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(out_per_block_u32),
        },
    );
    let meta2 = setup.let_(Ty::Usize, Rvalue::Use(cu(2)));
    let m_per_block_u32 = setup.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(metadata, meta2))));
    let m_per_block = setup.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(m_per_block_u32),
        },
    );
    // `x_groups` (`metadata[3]`): the number of tiles the fold-aware planner placed along the X grid
    // dim - `g = gy*x_groups + gx` recovers the flat block-major tile index `try_plan_contraction`
    // laid out, exactly as `gemv_lds`/`flash_region_prefill` recover their own flat index over a
    // `GroupY`-spilled grid. Unfolded, `x_groups` is the real tile count and `gy` is always 0, so `g`
    // is unchanged from a plain `GroupX` read.
    let meta3 = setup.let_(Ty::Usize, Rvalue::Use(cu(3)));
    let x_groups_u32 = setup.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(metadata, meta3))));
    let x_groups = setup.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(x_groups_u32),
        },
    );
    let g = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(gy), copy(x_groups));
    let g = bin(&mut setup, Ty::Usize, BinOp::Add, copy(g), copy(gx));

    // `rows`/`blocks` are re-derived from `out`'s own length (never baked, never a new metadata
    // word): `out` is exactly `[rows, out_per_block]` (card 542a's contract), so `rows = len(out) /
    // out_per_block`, `blocks = rows / m_per_block` (`OpKind::PackedContraction`'s own block-diagonal
    // split).
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
    let max_a_idx_raw = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(rows), copy(k));
    let max_a_idx = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Sub,
        copy(max_a_idx_raw),
        cu(1),
    );
    let k_minus_1 = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(k), cu(1));

    // Tile geometry: `tiles_col = ceil(out_per_block / ts)`, `kt_count = ceil(K / ts)`, both derived
    // at runtime from the metadata (never baked, so one body serves every shape of this format/op).
    let tiles_col_num = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Add,
        copy(out_per_block),
        cu(ts - 1),
    );
    let tiles_col = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(tiles_col_num),
        cu(ts),
    );
    let kt_count_num = bin(&mut setup, Ty::Usize, BinOp::Add, copy(k), cu(ts - 1));
    let kt_count = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(kt_count_num),
        cu(ts),
    );

    let tr = bin(&mut setup, Ty::Usize, BinOp::Div, copy(l), cu(ts));
    let tc = bin(&mut setup, Ty::Usize, BinOp::Rem, copy(l), cu(ts));

    // Card 658: `g` is laid out block-major - `tiles_per_block = ceil(m_per_block/ts) *
    // tiles_col` tiles per block, `blocks` groups of them - so `b` is this workgroup's fixed
    // block, never a per-lane derivation (the shared `Bs` tile this cooperative load stages is
    // therefore always one block's weight, regardless of whether `m_per_block` is a multiple of
    // `ts`).
    let tiles_row_per_block_num = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Add,
        copy(m_per_block),
        cu(ts - 1),
    );
    let tiles_row_per_block = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(tiles_row_per_block_num),
        cu(ts),
    );
    let tiles_per_block = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(tiles_row_per_block),
        copy(tiles_col),
    );
    let b = bin(
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
    let trow = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(tile_in_block),
        copy(tiles_col),
    );
    let tcol = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Rem,
        copy(tile_in_block),
        copy(tiles_col),
    );
    let trow_ts = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(trow), cu(ts));
    // `row_in_block` is this tile's row offset within block `b`'s own `m_per_block` rows (masked
    // against `m_per_block` directly below, never against the flattened `rows`): a padded tail
    // row (`row_in_block >= m_per_block`) must never alias into block `b + 1`'s rows.
    let row_in_block = bin(&mut setup, Ty::Usize, BinOp::Add, copy(trow_ts), copy(tr));
    let b_row_base = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(b),
        copy(m_per_block),
    );
    let row = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Add,
        copy(b_row_base),
        copy(row_in_block),
    );
    let tcol_ts = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(tcol), cu(ts));
    let col = bin(&mut setup, Ty::Usize, BinOp::Add, copy(tcol_ts), copy(tc));
    let weight_row_base = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(b),
        copy(out_per_block),
    );
    let weight_row_raw = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Add,
        copy(weight_row_base),
        copy(col),
    );
    let weight_row = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Min,
        copy(weight_row_raw),
        copy(max_weight_row),
    );
    let tr_ts = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(tr), cu(ts));

    let acc0 = setup.let_(
        Ty::F32,
        Rvalue::Use(Operand::Const(poot_kernel_ir::Constant::F32(0.0))),
    );
    let kt0 = setup.let_(Ty::Usize, Rvalue::Use(cu(0)));
    let mut bb2_stmts = setup.take();
    let acc_m = al.add(Ty::F32, true);
    let kt_m = al.add(Ty::Usize, true);
    bb2_stmts.push(Statement::Assign(
        Place::local(acc_m),
        Rvalue::Use(copy(acc0)),
    ));
    bb2_stmts.push(Statement::Assign(
        Place::local(kt_m),
        Rvalue::Use(copy(kt0)),
    ));
    let bb2 = BasicBlock {
        statements: bb2_stmts,
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };

    let mut guard_e = Emit::new(&mut al);
    let kt_lt = bin(
        &mut guard_e,
        Ty::Bool,
        BinOp::Lt,
        copy(kt_m),
        copy(kt_count),
    );
    let bb3 = BasicBlock {
        statements: guard_e.take(),
        terminator: guard(kt_lt, 7, 4),
    };

    // bb4: cooperative load. Each lane loads one activation element (`As[l]`, `l == tr*ts+tc`) and
    // decodes one weight element (`Bs[l]`, same slot, read back as `(k_local=tr, col_local=tc)` by the
    // dot loop below): clamp the pre-image indices (`row`/`a_k`, `weight_row`/`b_k`) into range first,
    // so every buffer read stays in bounds, then zero the tile-edge contribution via the 0/1 masks.
    let mut load = Emit::new(&mut al);
    let kt_ts = bin(&mut load, Ty::Usize, BinOp::Mul, copy(kt_m), cu(ts));
    let a_k = bin(&mut load, Ty::Usize, BinOp::Add, copy(kt_ts), copy(tc));
    let b_k = bin(&mut load, Ty::Usize, BinOp::Add, copy(kt_ts), copy(tr));

    // Masked against `m_per_block` (this tile's own block), never against the flattened `rows`: a
    // padded tail row from block `b`'s last row-tile must never read or contribute as if it were
    // block `b + 1`'s row (card 658).
    let rowok = bin(
        &mut load,
        Ty::Bool,
        BinOp::Lt,
        copy(row_in_block),
        copy(m_per_block),
    );
    let akok = bin(&mut load, Ty::Bool, BinOp::Lt, copy(a_k), copy(k));
    let colok = bin(
        &mut load,
        Ty::Bool,
        BinOp::Lt,
        copy(col),
        copy(out_per_block),
    );
    let bkok = bin(&mut load, Ty::Bool, BinOp::Lt, copy(b_k), copy(k));
    let row_mask = load.let_(
        Ty::F32,
        Rvalue::Cast {
            to: Ty::F32,
            operand: copy(rowok),
        },
    );
    let ak_mask = load.let_(
        Ty::F32,
        Rvalue::Cast {
            to: Ty::F32,
            operand: copy(akok),
        },
    );
    let col_mask = load.let_(
        Ty::F32,
        Rvalue::Cast {
            to: Ty::F32,
            operand: copy(colok),
        },
    );
    let bk_mask = load.let_(
        Ty::F32,
        Rvalue::Cast {
            to: Ty::F32,
            operand: copy(bkok),
        },
    );

    let a_row_off = bin(&mut load, Ty::Usize, BinOp::Mul, copy(row), copy(k));
    let a_idx_raw = bin(&mut load, Ty::Usize, BinOp::Add, copy(a_row_off), copy(a_k));
    let a_idx = bin(
        &mut load,
        Ty::Usize,
        BinOp::Min,
        copy(a_idx_raw),
        copy(max_a_idx),
    );
    let a_raw = load.let_(Ty::F32, Rvalue::Use(Operand::Copy(elem(activation, a_idx))));
    let a_masked1 = bin(&mut load, Ty::F32, BinOp::Mul, copy(a_raw), copy(row_mask));
    let a_val = bin(
        &mut load,
        Ty::F32,
        BinOp::Mul,
        copy(a_masked1),
        copy(ak_mask),
    );
    load.write_lds(0, copy(l), copy(a_val));

    let b_k_clamped = bin(&mut load, Ty::Usize, BinOp::Min, copy(b_k), copy(k_minus_1));
    let (block, i_in_block) = emit_block_address(&mut load, values, weight_row, b_k_clamped, k);
    let factors = plan.emit_factors(&mut load, words, block, i_in_block);
    let value = plan.emit_value(&mut load, words, block, i_in_block, &factors);
    let b_masked1 = bin(&mut load, Ty::F32, BinOp::Mul, copy(value), copy(col_mask));
    let b_val = bin(
        &mut load,
        Ty::F32,
        BinOp::Mul,
        copy(b_masked1),
        copy(bk_mask),
    );
    load.write_lds(1, copy(l), copy(b_val));

    let bb4 = BasicBlock {
        statements: load.take(),
        terminator: Terminator::Barrier {
            target: BlockId { index: 5 },
        },
    };

    // bb5: unrolled `tile`-wide dot over the two LDS tiles just staged, `As[row_local][k_local]`
    // (`row_local == tr`, array index `tr*ts+k_local`) times `Bs[k_local][col_local]` (`col_local ==
    // tc`, array index `k_local*ts+tc`).
    let mut dot = Emit::new(&mut al);
    for kk in 0..ts {
        let a_idx = bin(&mut dot, Ty::Usize, BinOp::Add, copy(tr_ts), cu(kk));
        let b_idx = bin(&mut dot, Ty::Usize, BinOp::Add, cu(kk * ts), copy(tc));
        let a_v = dot.read_lds(Ty::F32, 0, copy(a_idx));
        let b_v = dot.read_lds(Ty::F32, 1, copy(b_idx));
        let prod = bin(&mut dot, Ty::F32, BinOp::Mul, copy(a_v), copy(b_v));
        let acc_next = bin(&mut dot, Ty::F32, BinOp::Add, copy(acc_m), copy(prod));
        dot.assign(Place::local(acc_m), Rvalue::Use(copy(acc_next)));
    }
    let bb5 = BasicBlock {
        statements: dot.take(),
        terminator: Terminator::Goto {
            target: BlockId { index: 6 },
        },
    };

    let mut latch = Emit::new(&mut al);
    let kt_next = bin(&mut latch, Ty::Usize, BinOp::Add, copy(kt_m), cu(1));
    latch.assign(Place::local(kt_m), Rvalue::Use(copy(kt_next)));
    let bb6 = BasicBlock {
        statements: latch.take(),
        terminator: Terminator::Barrier {
            target: BlockId { index: 3 },
        },
    };

    let mut store_guard = Emit::new(&mut al);
    // Same `m_per_block`-relative mask as the load phase's `rowok` (card 658): a padded tail row
    // must not write into the next block's output row.
    let rowok2 = bin(
        &mut store_guard,
        Ty::Bool,
        BinOp::Lt,
        copy(row_in_block),
        copy(m_per_block),
    );
    let colok2 = bin(
        &mut store_guard,
        Ty::Bool,
        BinOp::Lt,
        copy(col),
        copy(out_per_block),
    );
    // Card 658: a folded grid's padding workgroups (`g >= tiles`, every fold that
    // is not an exact multiple of the X cap) compute `b >= blocks`. `row = b*m_per_block +
    // row_in_block` then lands past `rows` even though `row_in_block < m_per_block` alone still
    // holds (that mask only rules out straddling the NEXT block within a valid `b`, never an
    // out-of-range `b` itself) - an out-of-bounds store `rowok2`/`colok2` never caught. `row < rows`
    // closes it (and subsumes `rowok2` whenever `b < blocks`, kept anyway for the doc above).
    let rows_ok = bin(&mut store_guard, Ty::Bool, BinOp::Lt, copy(row), copy(rows));
    let storeok_rowcol = bin(
        &mut store_guard,
        Ty::Bool,
        BinOp::BitAnd,
        copy(rowok2),
        copy(colok2),
    );
    let storeok = bin(
        &mut store_guard,
        Ty::Bool,
        BinOp::BitAnd,
        copy(storeok_rowcol),
        copy(rows_ok),
    );
    let bb7 = BasicBlock {
        statements: store_guard.take(),
        terminator: guard(storeok, 9, 8),
    };

    let mut store = Emit::new(&mut al);
    let cidx_base = bin(
        &mut store,
        Ty::Usize,
        BinOp::Mul,
        copy(row),
        copy(out_per_block),
    );
    let cidx = bin(
        &mut store,
        Ty::Usize,
        BinOp::Add,
        copy(cidx_base),
        copy(col),
    );
    store.assign(elem(out, cidx), Rvalue::Use(copy(acc_m)));
    let bb8 = BasicBlock {
        statements: store.take(),
        terminator: Terminator::Return,
    };
    let bb9 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    // bb10: `gy = GroupY`, read here (appended, not inlined at bb0) so bb1..bb9's indices stay
    // stable (card 658 review F1/F2's fold-aware grid).
    let bb10 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gy),
            dim: IndexAxis::GroupY,
            target: BlockId { index: 1 },
        },
    };

    let mut body = Body::new(
        name,
        4,
        al.locals,
        vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9, bb10],
    );
    body.workgroup_size = [(ts * ts) as u32, 1, 1];
    body.workgroup_locals = vec![
        poot_kernel_ir::WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: (ts * ts) as u32,
        },
        poot_kernel_ir::WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: (ts * ts) as u32,
        },
    ];
    Ok(body)
}

/// [`contraction_tiled`]'s planar front end (card 542c): the same cooperative-load LDS tiling, but
/// `words` becomes `PlanarPlan::source_count()` parameters and the cooperative load's one-lane-one-
/// element decode reads `PlanarPlan` off `(weight_row, b_k_clamped)`. `out_total` is `weight_rows`
/// (`blocks * out_per_block`), already derived from `out`'s own length exactly as the block front
/// end's `contraction_tiled` computes it - the same fact, reused rather than rederived.
fn contraction_tiled_planar(
    name: &str,
    format: WeightFormat,
    tile: u32,
) -> Result<Body, KernelGenError> {
    if tile == 0 {
        return Err(KernelGenError::BelowMinimum {
            generator: "contraction_tiled_planar",
            what: "tile".to_string(),
            value: 0,
            min: 1,
        });
    }
    let descriptor = format.descriptor();
    let plan = PlanarPlan::of(&descriptor)?;
    let n = plan.source_count();
    let ts = tile as usize;

    let mut params = vec![ld(Ty::Unit, false), ld(slice_f32(false), false)];
    for _ in 0..n {
        params.push(ld(slice_dtype(Ty::U32, false), false));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_f32(true), true));
    let mut al = Alloc::new(params);
    let activation = local(1);
    let sources: Vec<Local> = (0..n as u32).map(|idx| local(2 + idx)).collect();
    let metadata = local(2 + n as u32);
    let out = local(3 + n as u32);
    let gx = al.add(Ty::Usize, false);
    let gy = al.add(Ty::Usize, false);
    let l = al.add(Ty::Usize, false);

    // Card 658 review F1/F2: see `contraction_tiled`'s identical doc - `gx`/`gy` are always read so
    // this body survives a folded dispatch grid; the GroupY read is appended as block 10.
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gx),
            dim: IndexAxis::GroupX,
            target: BlockId { index: 10 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(l),
            dim: IndexAxis::LocalX,
            target: BlockId { index: 2 },
        },
    };

    let mut setup = Emit::new(&mut al);
    let k = emit_read_meta(&mut setup, metadata, META_K);
    let meta1 = setup.let_(Ty::Usize, Rvalue::Use(cu(1)));
    let out_per_block_u32 = setup.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(metadata, meta1))));
    let out_per_block = setup.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(out_per_block_u32),
        },
    );
    let meta2 = setup.let_(Ty::Usize, Rvalue::Use(cu(2)));
    let m_per_block_u32 = setup.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(metadata, meta2))));
    let m_per_block = setup.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(m_per_block_u32),
        },
    );
    // `x_groups` (`metadata[3]`): see `contraction_tiled`'s identical doc.
    let meta3 = setup.let_(Ty::Usize, Rvalue::Use(cu(3)));
    let x_groups_u32 = setup.let_(Ty::U32, Rvalue::Use(Operand::Copy(elem(metadata, meta3))));
    let x_groups = setup.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(x_groups_u32),
        },
    );
    let g = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(gy), copy(x_groups));
    let g = bin(&mut setup, Ty::Usize, BinOp::Add, copy(g), copy(gx));

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
    let out_total = weight_rows;
    let max_weight_row = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(weight_rows), cu(1));
    let max_a_idx_raw = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(rows), copy(k));
    let max_a_idx = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Sub,
        copy(max_a_idx_raw),
        cu(1),
    );
    let k_minus_1 = bin(&mut setup, Ty::Usize, BinOp::Sub, copy(k), cu(1));

    let tiles_col_num = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Add,
        copy(out_per_block),
        cu(ts - 1),
    );
    let tiles_col = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(tiles_col_num),
        cu(ts),
    );
    let kt_count_num = bin(&mut setup, Ty::Usize, BinOp::Add, copy(k), cu(ts - 1));
    let kt_count = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(kt_count_num),
        cu(ts),
    );

    let tr = bin(&mut setup, Ty::Usize, BinOp::Div, copy(l), cu(ts));
    let tc = bin(&mut setup, Ty::Usize, BinOp::Rem, copy(l), cu(ts));

    // Card 658: `g` is laid out block-major (see `contraction_tiled`'s doc), so `b` is this
    // workgroup's fixed block, never a per-lane derivation.
    let tiles_row_per_block_num = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Add,
        copy(m_per_block),
        cu(ts - 1),
    );
    let tiles_row_per_block = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(tiles_row_per_block_num),
        cu(ts),
    );
    let tiles_per_block = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(tiles_row_per_block),
        copy(tiles_col),
    );
    let b = bin(
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
    let trow = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Div,
        copy(tile_in_block),
        copy(tiles_col),
    );
    let tcol = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Rem,
        copy(tile_in_block),
        copy(tiles_col),
    );
    let trow_ts = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(trow), cu(ts));
    // Masked against `m_per_block` directly below (never against the flattened `rows`): a padded
    // tail row must never alias into block `b + 1`'s rows.
    let row_in_block = bin(&mut setup, Ty::Usize, BinOp::Add, copy(trow_ts), copy(tr));
    let b_row_base = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(b),
        copy(m_per_block),
    );
    let row = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Add,
        copy(b_row_base),
        copy(row_in_block),
    );
    let tcol_ts = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(tcol), cu(ts));
    let col = bin(&mut setup, Ty::Usize, BinOp::Add, copy(tcol_ts), copy(tc));
    let weight_row_base = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Mul,
        copy(b),
        copy(out_per_block),
    );
    let weight_row_raw = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Add,
        copy(weight_row_base),
        copy(col),
    );
    let weight_row = bin(
        &mut setup,
        Ty::Usize,
        BinOp::Min,
        copy(weight_row_raw),
        copy(max_weight_row),
    );
    let tr_ts = bin(&mut setup, Ty::Usize, BinOp::Mul, copy(tr), cu(ts));

    let acc0 = setup.let_(
        Ty::F32,
        Rvalue::Use(Operand::Const(poot_kernel_ir::Constant::F32(0.0))),
    );
    let kt0 = setup.let_(Ty::Usize, Rvalue::Use(cu(0)));
    let mut bb2_stmts = setup.take();
    let acc_m = al.add(Ty::F32, true);
    let kt_m = al.add(Ty::Usize, true);
    bb2_stmts.push(Statement::Assign(
        Place::local(acc_m),
        Rvalue::Use(copy(acc0)),
    ));
    bb2_stmts.push(Statement::Assign(
        Place::local(kt_m),
        Rvalue::Use(copy(kt0)),
    ));
    let bb2 = BasicBlock {
        statements: bb2_stmts,
        terminator: Terminator::Goto {
            target: BlockId { index: 3 },
        },
    };

    let mut guard_e = Emit::new(&mut al);
    let kt_lt = bin(
        &mut guard_e,
        Ty::Bool,
        BinOp::Lt,
        copy(kt_m),
        copy(kt_count),
    );
    let bb3 = BasicBlock {
        statements: guard_e.take(),
        terminator: guard(kt_lt, 7, 4),
    };

    let mut load = Emit::new(&mut al);
    let kt_ts = bin(&mut load, Ty::Usize, BinOp::Mul, copy(kt_m), cu(ts));
    let a_k = bin(&mut load, Ty::Usize, BinOp::Add, copy(kt_ts), copy(tc));
    let b_k = bin(&mut load, Ty::Usize, BinOp::Add, copy(kt_ts), copy(tr));

    // Masked against `m_per_block` (this tile's own block), never against the flattened `rows`: a
    // padded tail row from block `b`'s last row-tile must never read or contribute as if it were
    // block `b + 1`'s row (card 658).
    let rowok = bin(
        &mut load,
        Ty::Bool,
        BinOp::Lt,
        copy(row_in_block),
        copy(m_per_block),
    );
    let akok = bin(&mut load, Ty::Bool, BinOp::Lt, copy(a_k), copy(k));
    let colok = bin(
        &mut load,
        Ty::Bool,
        BinOp::Lt,
        copy(col),
        copy(out_per_block),
    );
    let bkok = bin(&mut load, Ty::Bool, BinOp::Lt, copy(b_k), copy(k));
    let row_mask = load.let_(
        Ty::F32,
        Rvalue::Cast {
            to: Ty::F32,
            operand: copy(rowok),
        },
    );
    let ak_mask = load.let_(
        Ty::F32,
        Rvalue::Cast {
            to: Ty::F32,
            operand: copy(akok),
        },
    );
    let col_mask = load.let_(
        Ty::F32,
        Rvalue::Cast {
            to: Ty::F32,
            operand: copy(colok),
        },
    );
    let bk_mask = load.let_(
        Ty::F32,
        Rvalue::Cast {
            to: Ty::F32,
            operand: copy(bkok),
        },
    );

    let a_row_off = bin(&mut load, Ty::Usize, BinOp::Mul, copy(row), copy(k));
    let a_idx_raw = bin(&mut load, Ty::Usize, BinOp::Add, copy(a_row_off), copy(a_k));
    let a_idx = bin(
        &mut load,
        Ty::Usize,
        BinOp::Min,
        copy(a_idx_raw),
        copy(max_a_idx),
    );
    let a_raw = load.let_(Ty::F32, Rvalue::Use(Operand::Copy(elem(activation, a_idx))));
    let a_masked1 = bin(&mut load, Ty::F32, BinOp::Mul, copy(a_raw), copy(row_mask));
    let a_val = bin(
        &mut load,
        Ty::F32,
        BinOp::Mul,
        copy(a_masked1),
        copy(ak_mask),
    );
    load.write_lds(0, copy(l), copy(a_val));

    let b_k_clamped = bin(&mut load, Ty::Usize, BinOp::Min, copy(b_k), copy(k_minus_1));
    let factors = plan.emit_factors(&mut load, &sources, weight_row, b_k_clamped, k, out_total);
    let value = plan.emit_value(
        &mut load,
        &sources,
        weight_row,
        b_k_clamped,
        k,
        out_total,
        &factors,
    );
    let b_masked1 = bin(&mut load, Ty::F32, BinOp::Mul, copy(value), copy(col_mask));
    let b_val = bin(
        &mut load,
        Ty::F32,
        BinOp::Mul,
        copy(b_masked1),
        copy(bk_mask),
    );
    load.write_lds(1, copy(l), copy(b_val));

    let bb4 = BasicBlock {
        statements: load.take(),
        terminator: Terminator::Barrier {
            target: BlockId { index: 5 },
        },
    };

    let mut dot = Emit::new(&mut al);
    for kk in 0..ts {
        let a_idx = bin(&mut dot, Ty::Usize, BinOp::Add, copy(tr_ts), cu(kk));
        let b_idx = bin(&mut dot, Ty::Usize, BinOp::Add, cu(kk * ts), copy(tc));
        let a_v = dot.read_lds(Ty::F32, 0, copy(a_idx));
        let b_v = dot.read_lds(Ty::F32, 1, copy(b_idx));
        let prod = bin(&mut dot, Ty::F32, BinOp::Mul, copy(a_v), copy(b_v));
        let acc_next = bin(&mut dot, Ty::F32, BinOp::Add, copy(acc_m), copy(prod));
        dot.assign(Place::local(acc_m), Rvalue::Use(copy(acc_next)));
    }
    let bb5 = BasicBlock {
        statements: dot.take(),
        terminator: Terminator::Goto {
            target: BlockId { index: 6 },
        },
    };

    let mut latch = Emit::new(&mut al);
    let kt_next = bin(&mut latch, Ty::Usize, BinOp::Add, copy(kt_m), cu(1));
    latch.assign(Place::local(kt_m), Rvalue::Use(copy(kt_next)));
    let bb6 = BasicBlock {
        statements: latch.take(),
        terminator: Terminator::Barrier {
            target: BlockId { index: 3 },
        },
    };

    let mut store_guard = Emit::new(&mut al);
    // Same `m_per_block`-relative mask as the load phase's `rowok` (card 658): a padded tail row
    // must not write into the next block's output row.
    let rowok2 = bin(
        &mut store_guard,
        Ty::Bool,
        BinOp::Lt,
        copy(row_in_block),
        copy(m_per_block),
    );
    let colok2 = bin(
        &mut store_guard,
        Ty::Bool,
        BinOp::Lt,
        copy(col),
        copy(out_per_block),
    );
    // Card 658: a folded grid's padding workgroups (`g >= tiles`, every fold that
    // is not an exact multiple of the X cap) compute `b >= blocks`. `row = b*m_per_block +
    // row_in_block` then lands past `rows` even though `row_in_block < m_per_block` alone still
    // holds (that mask only rules out straddling the NEXT block within a valid `b`, never an
    // out-of-range `b` itself) - an out-of-bounds store `rowok2`/`colok2` never caught. `row < rows`
    // closes it (and subsumes `rowok2` whenever `b < blocks`, kept anyway for the doc above).
    let rows_ok = bin(&mut store_guard, Ty::Bool, BinOp::Lt, copy(row), copy(rows));
    let storeok_rowcol = bin(
        &mut store_guard,
        Ty::Bool,
        BinOp::BitAnd,
        copy(rowok2),
        copy(colok2),
    );
    let storeok = bin(
        &mut store_guard,
        Ty::Bool,
        BinOp::BitAnd,
        copy(storeok_rowcol),
        copy(rows_ok),
    );
    let bb7 = BasicBlock {
        statements: store_guard.take(),
        terminator: guard(storeok, 9, 8),
    };

    let mut store = Emit::new(&mut al);
    let cidx_base = bin(
        &mut store,
        Ty::Usize,
        BinOp::Mul,
        copy(row),
        copy(out_per_block),
    );
    let cidx = bin(
        &mut store,
        Ty::Usize,
        BinOp::Add,
        copy(cidx_base),
        copy(col),
    );
    store.assign(elem(out, cidx), Rvalue::Use(copy(acc_m)));
    let bb8 = BasicBlock {
        statements: store.take(),
        terminator: Terminator::Return,
    };
    let bb9 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    let bb10 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gy),
            dim: IndexAxis::GroupY,
            target: BlockId { index: 1 },
        },
    };

    let mut body = Body::new(
        name,
        3 + n as u32,
        al.locals,
        vec![bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9, bb10],
    );
    body.workgroup_size = [(ts * ts) as u32, 1, 1];
    body.workgroup_locals = vec![
        poot_kernel_ir::WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: (ts * ts) as u32,
        },
        poot_kernel_ir::WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: (ts * ts) as u32,
        },
    ];
    Ok(body)
}

#[cfg(test)]
mod tile_size_tests {
    use super::*;

    /// Card 658: `TileSize::new` no longer takes a `blocks` argument (the tiled body now serves any
    /// block count, `contraction_tiled`'s per-block grid doc) - a zero tile is the only refusal left.
    /// Mutation: dropping the `tile == 0` check turns this red.
    #[test]
    fn a_tiled_schedule_refuses_only_a_zero_tile() {
        assert_eq!(TileSize::new(8).map(TileSize::get), Ok(8));
        assert!(matches!(
            TileSize::new(0),
            Err(KernelGenError::BelowMinimum { .. })
        ));
        assert_eq!(
            format!("{:?}", Schedule::Tiled { tile: TileSize(8) }),
            "Tiled { tile: 8 }"
        );
    }
}
