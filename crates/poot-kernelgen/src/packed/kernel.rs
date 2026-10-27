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
use super::indexed::ExpertRows;
use super::planar::PlanarPlan;
use super::source::Source;
use crate::KernelGenError;
use crate::PackedRows;
use crate::contraction::{
    BoundEpilogue, Epilogue, GemvLaunch, RunLength, Schedule, TileSize, TiledBody, TiledLaunch,
    TiledWeight, gemv_body, tiled_body,
};
use crate::emit::{Emit, bin, copy, cu, emit_read_meta};
use crate::helpers::{Alloc, WeightLayout, elem, guard, ld, local, slice_dtype, slice_f32};
use crate::microtile::{LaneRows, MicroCols};

/// What one generated packed kernel computes (dquant.md 3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedKernelSpec {
    /// The weight format whose descriptor drives the decode.
    pub format: WeightFormat,
    /// The operation the kernel performs on the decoded weights.
    pub op: PackedKernelOp,
}

/// The operation a generated packed kernel performs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackedKernelOp {
    /// `out[r, :] = decode(ids[r], :)`: `Gather(axis 0)` over `PackedDequant`.
    RowGather,
}

/// `metadata[0]`: every generated body reads `K` (and further shape facts, by op) from this
/// parameter rather than baking a shape into the body (see the module docs).
const META_K: usize = 0;

/// The body of the packed kernel `spec` names, named `name`. A descriptor the decode emitter cannot state is a typed [`KernelGenError`].
pub fn packed_kernel(name: &str, spec: PackedKernelSpec) -> Result<Body, KernelGenError> {
    match spec.op {
        PackedKernelOp::RowGather => row_gather(name, spec.format),
    }
}

/// What the elements of one Gemv lane run (`count` aligned `K` elements of one weight row) share in a packed
/// decode: the fact the planner sizes the run from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedRunSharing {
    /// The decode addresses the run's block once, decodes its factors (scale, sub-scale, min, sub-min) once and
    /// reads its codes through shared 32-bit windows, when `count` divides `factor_run` (the aligned elements
    /// whose factor bits are all the same); a run that does not divide it decodes every element's factors.
    /// Block storage.
    FactorRun {
        /// The length of the aligned runs that share one set of factors.
        factor_run: usize,
    },
    /// The decode addresses every element's codes through its own chain (a float run of at most 32 code bits
    /// is one read), so a run shares at most its scale. Planar storage.
    Elementwise,
}

/// What a Gemv lane run of `format`'s decode shares ([`PackedRunSharing`]). A descriptor the decode cannot
/// state is the typed error its kernel would be.
pub fn packed_run_sharing(format: WeightFormat) -> Result<PackedRunSharing, KernelGenError> {
    let descriptor = format.descriptor();
    Ok(match descriptor.storage {
        Storage::Planar(_) => {
            PlanarPlan::of(&descriptor)?;
            PackedRunSharing::Elementwise
        }
        Storage::Blocks(_) => PackedRunSharing::FactorRun {
            factor_run: DecodePlan::of(&descriptor)?.factor_run(),
        },
    })
}

/// The packed operand load of a [`crate::ContractionSpec`]: `format`'s descriptor-driven decode served to
/// `schedule`'s loop nest, with `epilogue` applied to each column's sum before its store. The Gemv carries an
/// epilogue (its leaves ride between the weight sources and the metadata); the block-diagonal tiled schedule has
/// none yet, so an epilogue on it is a typed refusal. `rows` says how output rows meet weight rows: an
/// [`PackedRows::Indexed`] table is a Gemv (decode) or the register-tiled schedule inside an expert walk
/// (prefill), both with an epilogue.
pub(crate) fn packed_contraction(
    name: &str,
    format: WeightFormat,
    schedule: Schedule,
    epilogue: Option<&Epilogue>,
    rows: PackedRows,
) -> Result<Body, KernelGenError> {
    if let Some(epilogue) = epilogue {
        epilogue.require_f32("packed_contraction")?;
    }
    match schedule {
        Schedule::Gemv {
            width,
            cols,
            unroll,
            activation,
        } => contraction_gemv(
            name,
            format,
            GemvLaunch::new(width, cols, unroll, activation)?,
            epilogue,
            rows,
        ),
        Schedule::RegisterTiled {
            lanes,
            rows: lane_rows,
            cols,
            run,
        } if rows == PackedRows::Indexed => super::indexed::indexed_register_tiled(
            name,
            format,
            TiledLaunch::new(lanes, lane_rows, cols, run, WeightLayout::Nk)?,
            epilogue,
        ),
        Schedule::RegisterTiled { .. } if epilogue.is_some() => Err(KernelGenError::Unsupported {
            request: "packed_contraction",
            reason: "a packed RegisterTiled contraction has no epilogue",
        }),
        Schedule::RegisterTiled {
            lanes,
            rows,
            cols,
            run,
        } => contraction_register_tiled(name, format, lanes, rows, cols, run),
    }
}

/// `block = row*(k/values) + k_index/values`, `i_in_block = k_index % values`: the block-32 address
/// of logical `(row, k_index)` for a weight whose row holds `k` (total) logical values in blocks of
/// `values` (`DecodePlan::values`). Shared by every op ([`row_gather`],
/// [`contraction_gemv`], `contraction_register_tiled`) so the row-major block layout has one KIR
/// implementation, matching [`poot_quant::PackedWeight::source_shape`]'s host-side accounting.
pub(super) fn emit_block_address(
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

/// `out[r, :] = decode(ids[r], :)` for a `[R, K]` output, `ids` an `R`-length `I32` row index:
/// [`PackedKernelOp::RowGather`], the axis-0 `Gather` fused directly onto a `PackedDequant` producer
/// (GGUF token embeddings). Params `(words, ids, metadata, out)`, one thread per output
/// element; the row comes from `ids[r]`.
fn row_gather(name: &str, format: WeightFormat) -> Result<Body, KernelGenError> {
    let descriptor = format.descriptor();
    if let Storage::Planar(_) = descriptor.storage {
        return Err(KernelGenError::UnsupportedDescriptor {
            format,
            reason: "planar RowGather (only Contraction is admitted for planar storage; \
                     no production consumer gathers planar-format rows today)",
        });
    }
    let plan = DecodePlan::of(&descriptor)?;
    let values = plan.values();

    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_dtype(Ty::I32, false), false),
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
    let row_id = emit.let_(Ty::I32, Rvalue::Use(Operand::Copy(elem(ids, r))));
    let row = emit.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(row_id),
        },
    );
    let (block, i_in_block) = emit_block_address(&mut emit, values, row, kk, k);

    let factors = plan.emit_factors(&mut emit, Source::plain(words), block, i_in_block);
    let value = plan.emit_value(&mut emit, Source::plain(words), block, i_in_block, &factors);
    emit.assign(elem(out, i), Rvalue::Use(copy(value)));

    let bb2 = BasicBlock {
        statements: emit.take(),
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    let mut body = Body::new(name, 4, al.locals, vec![bb0, bb1, bb2, bb3]);
    body.constant_tables = plan.constant_tables();
    Ok(body)
}

/// `out[col_flat] = sum over k of activation[b, k] * decode(col_flat, k)`, `M == 1`:
/// [`Schedule::Gemv`] over the block-storage front end. `col_flat` ranges over `blocks *
/// out_per_block` (the flattened `[blocks, out_per_block]` output, which is also the weight row
/// index; `blocks == 1` is the ordinary dense GEMV) and `b = col_flat / out_per_block`. Params
/// `(activation, words, metadata, out)` - activation first, matching `OpKind::PackedContraction`'s
/// own operand order (card 542a production wiring, D9). `metadata` is `[K, out_per_block]`;
/// `activation` is `[blocks, K]` (one row per block, `M == 1`). With an `epilogue` its leaves are
/// parameters between `words` and `metadata`, and it runs over each column's sum before the store.
///
/// [`PackedRows::Indexed`] is the same loop over a stacked table: params `(activation, ids, words, ..)`,
/// flat column `m * N + n` (`N = out_per_block`) reads weight row `n` of the slab of expert `ids[m]`
/// ([`ExpertRows`]), and `metadata` is `[K, N, E, slab_words]`.
fn contraction_gemv(
    name: &str,
    format: WeightFormat,
    launch: GemvLaunch,
    epilogue: Option<&Epilogue>,
    rows: PackedRows,
) -> Result<Body, KernelGenError> {
    let descriptor = format.descriptor();
    if let Storage::Planar(_) = descriptor.storage {
        return contraction_gemv_planar(name, format, launch, epilogue, rows);
    }
    let plan = DecodePlan::of(&descriptor)?;
    let leaf_count = epilogue.map_or(0, Epilogue::leaf_count);
    let lanes = match epilogue {
        Some(epilogue) => epilogue.lanes("packed_contraction")?,
        None => Vec::new(),
    };
    let lead = rows.leading_params();
    let mut params = vec![ld(Ty::Unit, false), ld(slice_f32(false), false)];
    if rows == PackedRows::Indexed {
        params.push(ld(slice_dtype(Ty::I32, false), false));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    if let Some(epilogue) = epilogue {
        params.extend(lanes.iter().map(|&lane| epilogue.leaf_param(lane)));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_f32(true), true));
    let al = Alloc::new(params);
    let activation = local(1);
    let words = local(lead);
    let leaves: Vec<Local> = (0..leaf_count)
        .map(|j| local(lead + 1 + j as u32))
        .collect();
    let (metadata, out) = (
        local(lead + 1 + leaf_count as u32),
        local(lead + 2 + leaf_count as u32),
    );
    let selection = (rows == PackedRows::Indexed).then(|| ExpertRows {
        ids: local(2),
        metadata,
    });
    let decode_run = |e: &mut Emit, row: Local, k0: Local, count: usize, k: Local, _: Local| {
        let (row, src, poison) = match selection {
            Some(selection) => {
                let selected = selection.select(e, row, 1);
                (
                    selected.weight_row,
                    selected.source(0, words),
                    Some(selected.poison),
                )
            }
            None => (row, Source::plain(words), None),
        };
        let decoded = plan.emit_run(e, src, row, k0, count, k);
        poisoned(e, decoded, poison)
    };
    let mut body = gemv_body(
        name,
        al,
        lead + 2 + leaf_count as u32,
        [activation, metadata, out],
        launch,
        epilogue.map(|epilogue| BoundEpilogue {
            epilogue,
            leaves: &leaves,
            lanes: &lanes,
        }),
        decode_run,
    );
    body.constant_tables = plan.constant_tables();
    Ok(body)
}

/// `values` with `poison` (`0.0` or NaN, [`ExpertRows`]) added to the first, so an invalid expert id makes the
/// whole column NaN; `values` unchanged for a contraction with no row selection.
fn poisoned(e: &mut Emit, mut values: Vec<Local>, poison: Option<Local>) -> Vec<Local> {
    if let (Some(poison), Some(first)) = (poison, values.first_mut()) {
        *first = bin(e, Ty::F32, BinOp::Add, copy(*first), copy(poison));
    }
    values
}

/// [`contraction_gemv`]'s planar front end (card 542c): the same schedule, `words` becomes
/// `PlanarPlan::source_count()` parameters, the per-element decode reads `PlanarPlan` off
/// `(weight_row, k)`. `out_total` is `len(out)` (`M == 1` gives `rows == blocks`, so `out`'s
/// `[blocks, out_per_block]` flattens to exactly the weight's true row count); an indexed table's `out`
/// is `[M, N]`, so its decode reads `N` from the metadata instead, and each source is read relative to the
/// selected expert's slab.
fn contraction_gemv_planar(
    name: &str,
    format: WeightFormat,
    launch: GemvLaunch,
    epilogue: Option<&Epilogue>,
    rows: PackedRows,
) -> Result<Body, KernelGenError> {
    let descriptor = format.descriptor();
    let plan = PlanarPlan::of(&descriptor)?;
    let n = plan.source_count();
    let leaf_count = epilogue.map_or(0, Epilogue::leaf_count);
    let lanes = match epilogue {
        Some(epilogue) => epilogue.lanes("packed_contraction")?,
        None => Vec::new(),
    };
    let lead = rows.leading_params();
    let mut params = vec![ld(Ty::Unit, false), ld(slice_f32(false), false)];
    if rows == PackedRows::Indexed {
        params.push(ld(slice_dtype(Ty::I32, false), false));
    }
    for _ in 0..n {
        params.push(ld(slice_dtype(Ty::U32, false), false));
    }
    if let Some(epilogue) = epilogue {
        params.extend(lanes.iter().map(|&lane| epilogue.leaf_param(lane)));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_f32(true), true));
    let al = Alloc::new(params);
    let activation = local(1);
    let words: Vec<Local> = (0..n as u32).map(|idx| local(lead + idx)).collect();
    let leaves: Vec<Local> = (0..leaf_count)
        .map(|j| local(lead + (n + j) as u32))
        .collect();
    let metadata = local(lead + (n + leaf_count) as u32);
    let out = local(lead + 1 + (n + leaf_count) as u32);
    let selection = (rows == PackedRows::Indexed).then(|| ExpertRows {
        ids: local(2),
        metadata,
    });
    let decode_run =
        |e: &mut Emit, row: Local, k0: Local, count: usize, k: Local, out_total: Local| {
            let (row, out_total, sources, poison) = match selection {
                Some(selection) => {
                    let selected = selection.select(e, row, n);
                    let sources: Vec<Source> = words
                        .iter()
                        .enumerate()
                        .map(|(index, &words)| selected.source(index, words))
                        .collect();
                    (
                        selected.weight_row,
                        selected.out_rows,
                        sources,
                        Some(selected.poison),
                    )
                }
                None => (
                    row,
                    out_total,
                    words.iter().copied().map(Source::plain).collect(),
                    None,
                ),
            };
            let decoded: Vec<Local> = (0..count)
                .map(|j| {
                    let k_index = bin(e, Ty::Usize, BinOp::Add, copy(k0), cu(j));
                    let factors = plan.emit_factors(e, &sources, row, k_index, k, out_total);
                    plan.emit_value(e, &sources, row, k_index, k, out_total, &factors)
                })
                .collect();
            poisoned(e, decoded, poison)
        };
    Ok(gemv_body(
        name,
        al,
        lead + 1 + (n + leaf_count) as u32,
        [activation, metadata, out],
        launch,
        epilogue.map(|epilogue| BoundEpilogue {
            epilogue,
            leaves: &leaves,
            lanes: &lanes,
        }),
        decode_run,
    ))
}

/// `out[row, col] = sum over k of activation[row, k] * decode(weight_row(row, col), k)`, `M > 1`:
/// [`Schedule::RegisterTiled`], the block-diagonal loop nest of [`tiled_body`] over `format`'s
/// descriptor-driven decode. A workgroup cooperatively stages one activation tile and one decoded weight tile
/// into LDS per K round (each lane decodes `slots` weight elements through [`DecodePlan`] or [`PlanarPlan`],
/// never a per-format unpack) and every lane accumulates a register tile of the output; the tile geometry,
/// masks and fold-aware grid are [`tiled_body`]'s. Params `(activation, words.., metadata, out)` - activation
/// first, matching `OpKind::PackedContraction`'s own operand order; a block-storage format has one `words`
/// source, a planar one `PlanarPlan::source_count()`.
fn contraction_register_tiled(
    name: &str,
    format: WeightFormat,
    lanes: TileSize,
    rows: LaneRows,
    cols: MicroCols,
    run: RunLength,
) -> Result<Body, KernelGenError> {
    let launch = TiledLaunch::new(lanes, rows, cols, run, WeightLayout::Nk)?;
    let descriptor = format.descriptor();
    let (source_count, block_plan, planar_plan) = match descriptor.storage {
        Storage::Planar(_) => {
            let plan = PlanarPlan::of(&descriptor)?;
            (plan.source_count(), None, Some(plan))
        }
        _ => (1, Some(DecodePlan::of(&descriptor)?), None),
    };
    let mut params = vec![ld(Ty::Unit, false), ld(slice_f32(false), false)];
    for _ in 0..source_count {
        params.push(ld(slice_dtype(Ty::U32, false), false));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_f32(true), true));
    let sources: Vec<Source> = (0..source_count as u32)
        .map(|idx| Source::plain(local(2 + idx)))
        .collect();
    let spec = TiledBody {
        name,
        params,
        activation: local(1),
        metadata: local(2 + source_count as u32),
        out: local(3 + source_count as u32),
        epilogue: None,
        walk: None,
    };
    let mut body = match (&block_plan, &planar_plan) {
        (Some(plan), _) => tiled_body(
            spec,
            launch,
            TiledWeight::Rows(&mut |e, _step, row, k0, count, k, _weight_rows| {
                plan.emit_run(e, sources[0], row, k0, count, k)
            }),
        ),
        (None, Some(plan)) => tiled_body(
            spec,
            launch,
            TiledWeight::Rows(&mut |e, _step, row, k0, count, k, weight_rows| {
                plan.emit_run(e, &sources, row, k0, count, k, weight_rows)
            }),
        ),
        (None, None) => unreachable!("a storage is block or planar"),
    };
    if let Some(plan) = &block_plan {
        body.constant_tables = plan.constant_tables();
    }
    Ok(body)
}

#[cfg(test)]
mod tile_size_tests {
    use super::*;

    /// Card 658: `TileSize::new` no longer takes a `blocks` argument (the tiled body now serves any
    /// block count, `tiled_body`'s per-block grid doc) - a zero tile is the only refusal left.
    /// Mutation: dropping the `tile == 0` check turns this red.
    #[test]
    fn a_tiled_schedule_refuses_only_a_zero_tile() {
        assert_eq!(TileSize::new(8).map(TileSize::get), Ok(8));
        assert!(matches!(
            TileSize::new(0),
            Err(KernelGenError::BelowMinimum { .. })
        ));
        assert_eq!(format!("{:?}", TileSize(8)), "8");
    }
}
