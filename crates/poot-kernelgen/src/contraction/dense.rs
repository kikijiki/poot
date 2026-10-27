//! The dense operand load: a contraction weight held as stored (f32, or BF16/F16 checkpoint bytes packed
//! two per `u32` word), read in the orientation the request names.

use poot_kernel_ir::{BinOp, Body, Local, Ty};
use poot_target::BufferStorage;

use super::Schedule;
use super::epilogue::{BoundEpilogue, Epilogue};
use super::gemv::{GemvLaunch, gemv_body};
use super::tiled::{TiledBody, TiledLaunch, TiledWeight, tiled_body};
use crate::KernelGenError;
use crate::emit::{Emit, bin, copy};
use crate::helpers::{Alloc, WeightLane, WeightLayout, ld, local, slice_dtype, slice_f32};

/// The weight of a dense `M > 1` contraction: its orientation and its planned storage. Only the checkpoint's
/// `[N, K]` order reads a narrow (packed) weight; a `[K, N]` weight (the operand of a traced `MatMul`) is f32, so a
/// `[K, N]` weight in a packed lane cannot be written down.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DenseTiledWeight {
    /// `[N, K]`, `K` contiguous, in this storage: `BufferStorage::f32()`, `f16_packed()` or `bf16_packed()`.
    Nk(BufferStorage),
    /// `[K, N]` f32, one `[K, N]` slab per block.
    KnF32,
}

impl DenseTiledWeight {
    /// The orientation the tiled body stages.
    pub(crate) fn layout(self) -> WeightLayout {
        match self {
            Self::Nk(_) => WeightLayout::Nk,
            Self::KnF32 => WeightLayout::Kn,
        }
    }

    /// The lane the weight is read through.
    fn lane(self, generator: &'static str) -> Result<WeightLane, KernelGenError> {
        match self {
            Self::Nk(storage) => WeightLane::of(generator, storage),
            Self::KnF32 => Ok(WeightLane::F32),
        }
    }
}

/// The body of a dense `M == 1` contraction under `schedule` (a [`Schedule::Gemv`]) with contraction depth `k`,
/// params `(activation, weight, epilogue leaves.., metadata, out)`, `metadata = [K, N]` (the body reads both per
/// call; `k` only fixes whether a packed weight's row runs start on a word boundary, so the planner passes the
/// same `K` it generated for). `epilogue` runs over each column's sum before its store (f32 only: the dense
/// Gemv's output is f32). The shared Gemv loop nest over an `[N, K]` weight, each lane reading runs of its output
/// column's contiguous weight row through [`WeightLane::emit_run`]. The weight is `[N, K]` by construction: a
/// `[K, N]` weight would be read at a stride of `N` elements per lane, the access pattern card 035 removed, so
/// the request cannot name that orientation. An `M > 1` schedule is [`dense_tiled_contraction`]'s.
pub(crate) fn dense_contraction(
    name: &str,
    weight: BufferStorage,
    schedule: Schedule,
    k: usize,
    epilogue: Option<&Epilogue>,
) -> Result<Body, KernelGenError> {
    let lane = WeightLane::of("dense_contraction", weight)?;
    if let Some(epilogue) = epilogue {
        epilogue.require_f32("dense_contraction")?;
    }
    match schedule {
        Schedule::Gemv {
            width,
            cols,
            unroll,
            activation,
        } => dense_gemv(
            name,
            lane,
            GemvLaunch::new(width, cols, unroll, activation)?,
            k,
            epilogue,
        ),
        Schedule::RegisterTiled { .. } => Err(KernelGenError::Unsupported {
            request: "dense_contraction",
            reason: "a dense M > 1 contraction is the register-tiled DenseTiled request",
        }),
    }
}

/// `out[row, n] = sum over k of activation[row, k] * weight(n, k)`, `M > 1`: [`Schedule::RegisterTiled`], the
/// block-diagonal loop nest of [`tiled_body`] over a dense weight, with `epilogue` (f32) applied to each output
/// before its store. Params `(activation, weight, epilogue leaves.., metadata, out)`, `metadata = [K,
/// out_per_block, m_per_block, x_groups, tile_offset]`. An `[N, K]` weight stages runs of its rows through
/// [`WeightLane::emit_run`]: row `n` starts at element `n * K`, so with `K` a multiple of the lane's elements per
/// word every run (a multiple of the word too) starts on a word boundary and loads whole words. A `[K, N]` weight
/// stages its f32 elements along `N`.
pub(crate) fn dense_tiled_contraction(
    name: &str,
    weight: DenseTiledWeight,
    schedule: Schedule,
    k: usize,
    epilogue: Option<&Epilogue>,
) -> Result<Body, KernelGenError> {
    const GENERATOR: &str = "dense_tiled_contraction";
    let lane = weight.lane(GENERATOR)?;
    let Schedule::RegisterTiled {
        lanes,
        rows,
        cols,
        run,
    } = schedule
    else {
        return Err(KernelGenError::Unsupported {
            request: GENERATOR,
            reason: "a dense M > 1 contraction runs the RegisterTiled schedule",
        });
    };
    let launch = TiledLaunch::new(lanes, rows, cols, run, weight.layout())?;
    let leaf_count = epilogue.map_or(0, Epilogue::leaf_count);
    let leaf_lanes = match epilogue {
        Some(epilogue) => {
            epilogue.require_f32(GENERATOR)?;
            epilogue.lanes(GENERATOR)?
        }
        None => Vec::new(),
    };
    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        lane.param(),
    ];
    if let Some(epilogue) = epilogue {
        params.extend(leaf_lanes.iter().map(|&lane| epilogue.leaf_param(lane)));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_f32(true), true));
    let (activation, weight_buffer) = (local(1), local(2));
    let leaves: Vec<Local> = (0..leaf_count).map(|j| local(3 + j as u32)).collect();
    let spec = TiledBody {
        name,
        params,
        activation,
        metadata: local(3 + leaf_count as u32),
        out: local(4 + leaf_count as u32),
        epilogue: epilogue.map(|epilogue| BoundEpilogue {
            epilogue,
            leaves: &leaves,
            lanes: &leaf_lanes,
        }),
        walk: None,
    };
    let rows_aligned = k.is_multiple_of(lane.per_word());
    let word_aligned = rows_aligned && (run.get() as usize).is_multiple_of(lane.per_word());
    Ok(match weight {
        DenseTiledWeight::Nk(_) => tiled_body(
            spec,
            launch,
            TiledWeight::Rows(&mut |e, _step, row, k0, count, k, _weight_rows| {
                let first = row_run_start(e, row, k0, k);
                lane.emit_run(e, weight_buffer, first, count, word_aligned)
            }),
        ),
        DenseTiledWeight::KnF32 => tiled_body(
            spec,
            launch,
            TiledWeight::Columns {
                weight: weight_buffer,
                lane,
            },
        ),
    })
}

/// The element index of value `k0` of weight row `row` in a row-major `[_, k]` weight.
fn row_run_start(e: &mut Emit, row: Local, k0: Local, k: Local) -> Local {
    let row_base = bin(e, Ty::Usize, BinOp::Mul, copy(row), copy(k));
    bin(e, Ty::Usize, BinOp::Add, copy(row_base), copy(k0))
}

/// `out[n] = sum over k of activation[b, k] * weight[n, k]` over an `[N, K]` weight in `lane`'s storage: the
/// Gemv loop nest with a dense run read. Weight row `n` starts at element `n * K`, so with `K` a multiple of
/// the lane's elements per word every run of a whole number of words starts on a word boundary and loads
/// whole words; a run that does not (an odd `K`, or the one-element tail) reads element by element.
fn dense_gemv(
    name: &str,
    lane: WeightLane,
    launch: GemvLaunch,
    k: usize,
    epilogue: Option<&Epilogue>,
) -> Result<Body, KernelGenError> {
    let leaf_count = epilogue.map_or(0, Epilogue::leaf_count);
    let lanes = match epilogue {
        Some(epilogue) => epilogue.lanes("dense_contraction")?,
        None => Vec::new(),
    };
    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        lane.param(),
    ];
    if let Some(epilogue) = epilogue {
        params.extend(lanes.iter().map(|&lane| epilogue.leaf_param(lane)));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_f32(true), true));
    let al = Alloc::new(params);
    let (activation, weight) = (local(1), local(2));
    let leaves: Vec<Local> = (0..leaf_count).map(|j| local(3 + j as u32)).collect();
    let (metadata, out) = (local(3 + leaf_count as u32), local(4 + leaf_count as u32));
    let per_word = lane.per_word();
    let rows_aligned = k.is_multiple_of(per_word);
    // `gemv_body`'s shared run-read signature: only the planar packed front end reads `out_total`.
    let read_run =
        |e: &mut Emit, row: Local, k0: Local, count: usize, k: Local, _out_total: Local| {
            let first = row_run_start(e, row, k0, k);
            let word_aligned = rows_aligned && count.is_multiple_of(per_word);
            lane.emit_run(e, weight, first, count, word_aligned)
        };
    let epilogue = epilogue.map(|epilogue| BoundEpilogue {
        epilogue,
        leaves: &leaves,
        lanes: &lanes,
    });
    Ok(gemv_body(
        name,
        al,
        3 + leaf_count as u32 + 1,
        [activation, metadata, out],
        launch,
        epilogue,
        read_run,
    ))
}

#[cfg(test)]
mod tests {
    use poot_kernel_ir::interp::{Buffer, run};
    use poot_target::{BufferStorage, DeviceCaps};

    use super::{DenseTiledWeight, dense_tiled_contraction};
    use crate::contraction::RegisterTiling;
    use crate::helpers::WeightLayout;

    /// Values in `[-1, 1)` on a fixed xorshift stream, rounded to what a BF16 and an F16 store both hold exactly
    /// (eight mantissa bits, small exponents), so every storage reads the same numbers.
    fn values(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) % 256) as f32 / 128.0 - 1.0
            })
            .collect()
    }

    /// `weight` as its storage holds it: f32 words, or two 16-bit halves per `u32` word (element `i` in word
    /// `i / 2` at bit `(i % 2) * 16`).
    fn stored(weight: DenseTiledWeight, values: &[f32]) -> Buffer {
        let halves = |encode: fn(f32) -> u16| -> Buffer {
            let words: Vec<u32> = values
                .chunks(2)
                .map(|pair| {
                    u32::from(encode(pair[0]))
                        | (u32::from(pair.get(1).map_or(0, |&v| encode(v))) << 16)
                })
                .collect();
            Buffer::from_u32s(&words)
        };
        match weight {
            DenseTiledWeight::Nk(storage) if storage == BufferStorage::bf16_packed() => {
                halves(|v| (v.to_bits() >> 16) as u16)
            }
            DenseTiledWeight::Nk(storage) if storage == BufferStorage::f16_packed() => {
                halves(poot_quant::scalar::f32_to_f16)
            }
            _ => Buffer::from_f32s(values),
        }
    }

    /// Every tiling the planner weighs, over every dense weight storage in both orientations: two blocks of 37
    /// rows by 41 columns (ragged in both, a block-diagonal product whose blocks read their own weight rows or
    /// slab), with a `K` that stages as whole four-value runs (20) and one whose runs are single values with a
    /// tail past the last 16-value stage (17), match a host dot product. Mutation: in `tiled_body`'s `[K, N]`
    /// staging, drop the block's slab from `source`; every `KnF32` case goes red at the second block's rows.
    #[test]
    fn the_dense_tiled_body_matches_a_host_product_at_every_planned_tiling() {
        let caps = DeviceCaps::rocm_default();
        let (blocks, m, n) = (2usize, 37usize, 41usize);
        for k in [20usize, 17] {
            let activation = values(blocks * m * k, 1);
            let weights = values(blocks * n * k, 2);
            for weight in [
                DenseTiledWeight::Nk(BufferStorage::f32()),
                DenseTiledWeight::Nk(BufferStorage::bf16_packed()),
                DenseTiledWeight::Nk(BufferStorage::f16_packed()),
                DenseTiledWeight::KnF32,
            ] {
                // Weight element (block b, column c, depth d) in the buffer's own order.
                let at = |b: usize, c: usize, d: usize| match weight.layout() {
                    WeightLayout::Nk => weights[(b * n + c) * k + d],
                    WeightLayout::Kn => weights[(b * k + d) * n + c],
                };
                let want: Vec<f32> = (0..blocks * m * n)
                    .map(|i| {
                        let (row, c) = (i / n, i % n);
                        let b = row / m;
                        (0..k).map(|d| activation[row * k + d] * at(b, c, d)).sum()
                    })
                    .collect();
                for tiling in RegisterTiling::candidates(k, weight.layout(), &caps) {
                    let body = dense_tiled_contraction("t", weight, tiling.schedule(), k, None)
                        .expect("a planned tiling generates");
                    let (tile_rows, tile_cols) = tiling.output_tile();
                    let tiles = blocks * m.div_ceil(tile_rows) * n.div_ceil(tile_cols);
                    let meta = [k as u32, n as u32, m as u32, tiles as u32, 0];
                    let mut buffers = vec![
                        Buffer::from_f32s(&activation),
                        stored(weight, &weights),
                        Buffer::from_u32s(&meta),
                        Buffer::from_f32s(&vec![0.0; blocks * m * n]),
                    ];
                    run(&body, [tiles as u32, 1, 1], &mut buffers).expect("interpreter");
                    let got = buffers.pop().unwrap().to_f32s().unwrap();
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
                            "{weight:?} K={k} {tiling:?} [{}, {}]: {g} vs {w}",
                            i / n,
                            i % n
                        );
                    }
                }
            }
        }
    }
}
