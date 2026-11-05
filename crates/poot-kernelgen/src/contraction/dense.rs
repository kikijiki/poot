//! The dense operand load: a contraction weight held as stored (f32, or BF16/F16 checkpoint bytes packed
//! two per `u32` word), read in the orientation the request names.

use poot_kernel_ir::{BinOp, Body, Local, Ty};
use poot_target::{BufferStorage, RoundingConversions};

use super::Schedule;
use super::epilogue::{BoundEpilogue, Epilogue};
use super::gemv::{GemvLaunch, gemv_body};
use super::tiled::{TiledBody, TiledExtents, TiledLaunch, TiledWeight, tiled_body};
use crate::KernelGenError;
use crate::emit::{Emit, bin, copy};
use crate::helpers::{Alloc, WeightLane, WeightLayout, ld, local, slice_dtype, slice_f32};
use crate::precision::{ProductRead, stored_ty};
use poot_graph_ir::{FragmentDtype, OperandPrecision};

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

    /// The bits one weight element occupies in global memory ([`crate::TiledShape::weight_bits`]): 16 for a packed
    /// half, 32 for an f32 word.
    pub fn bits(self) -> u32 {
        match self {
            Self::Nk(storage) => crate::dense_value_bits(storage),
            Self::KnF32 => crate::dense_value_bits(BufferStorage::f32()),
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
    precision: OperandPrecision,
    conversions: RoundingConversions,
) -> Result<Body, KernelGenError> {
    let lane = WeightLane::of("dense_contraction", weight)?;
    if let Some(epilogue) = epilogue {
        epilogue.require_f32("dense_contraction")?;
        epilogue.refuse_narrow_store("dense_contraction")?;
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
            ProductRead::dense(precision, lane, conversions),
        ),
        Schedule::RegisterTiled { .. } | Schedule::FragmentTiled { .. } => {
            Err(KernelGenError::Unsupported {
                request: "dense_contraction",
                reason: "a dense M > 1 contraction is the register-tiled DenseTiled request",
            })
        }
    }
}

/// `out[row, n] = sum over k of activation[row, k] * weight(n, k)`, `M > 1`: [`Schedule::RegisterTiled`], the
/// block-diagonal loop nest of [`tiled_body`] over a dense weight, with `epilogue` (f32) applied to each output
/// before its store. Params `(activation, weight, epilogue leaves.., metadata, out)`, `metadata = [K,
/// out_per_block, m_per_block, x_groups, tile_offset]`, of which `extents` are the first two. An `[N, K]` weight
/// stages runs of its rows through [`WeightLane::emit_run`]: row `n` starts at element `n * K`, so with `K` a
/// multiple of the lane's elements per word every run (a multiple of the word too) starts on a word boundary and
/// loads whole words. A `[K, N]` weight stages its f32 elements in blocks along `N` ([`tiled_body`]).
pub(crate) fn dense_tiled_contraction(
    name: &str,
    weight: DenseTiledWeight,
    schedule: Schedule,
    extents: TiledExtents,
    epilogue: Option<&Epilogue>,
    read: TiledRead,
) -> Result<Body, KernelGenError> {
    let TiledRead {
        precision,
        activation_stored,
        conversions,
    } = read;
    const GENERATOR: &str = "dense_tiled_contraction";
    let lane = weight.lane(GENERATOR)?;
    let launch = TiledLaunch::of(schedule, weight.layout(), precision, extents)?;
    let k = extents.k;
    let leaf_count = epilogue.map_or(0, Epilogue::leaf_count);
    let leaf_lanes = match epilogue {
        Some(epilogue) => {
            epilogue.require_f32(GENERATOR)?;
            epilogue.lanes(GENERATOR)?
        }
        None => Vec::new(),
    };
    let store = epilogue.and_then(Epilogue::store);
    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(stored_ty(activation_stored), false), false),
        lane.param(),
    ];
    if let Some(epilogue) = epilogue {
        params.extend(leaf_lanes.iter().map(|&lane| epilogue.leaf_param(lane)));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_dtype(stored_ty(store), true), true));
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
        grouped: None,
        read: ProductRead {
            activation_stored,
            ..ProductRead::dense(precision, lane, conversions)
        },
    };
    let rows_aligned = k.is_multiple_of(lane.per_word());
    let word_aligned = rows_aligned && launch.run().is_multiple_of(lane.per_word());
    Ok(match weight {
        DenseTiledWeight::Nk(_) => tiled_body(
            spec,
            launch,
            TiledWeight::Rows(&mut |e, _step, row, k0, count, k, _weight_rows| {
                let first = row_run_start(e, row, k0, k);
                lane.emit_run(e, conversions, weight_buffer, first, count, word_aligned)
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

/// How a dense tiled contraction reads its activation (POOT-1448): the declared precision of its products, the
/// fragment dtype the activation buffer holds its values in, if the planner stored it narrow (`None`: f32), and how
/// the device rounds to a fragment dtype (POOT-1458). An activation stored in the fragment dtype of a narrowed read
/// is exact in it and enters unrounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TiledRead {
    pub(crate) precision: OperandPrecision,
    pub(crate) activation_stored: Option<FragmentDtype>,
    pub(crate) conversions: RoundingConversions,
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
    read: ProductRead,
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
            lane.emit_run(e, read.conversions, weight, first, count, word_aligned)
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
        read,
        epilogue,
        read_run,
    ))
}

#[cfg(test)]
mod tests {
    use poot_kernel_ir::interp::{Buffer, run};
    use poot_target::{BufferStorage, DeviceCaps};

    use super::{DenseTiledWeight, dense_tiled_contraction};
    use crate::StagedTerms;
    use crate::contraction::{RegisterTiling, TiledShape};
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
    /// slab), with a `K` that stages as whole four-value runs (20), one whose runs are single values with a tail
    /// past the last 16-value round (17) and one of whole rounds (32: no tail mask, the round the pipeline issues
    /// past the last clamped), match a host dot product. Mutation: in `tiled_body`'s `[K, N]` staging, drop the
    /// block's slab from `source`; every `KnF32` case goes red at the second block's rows. Mutation: drop the
    /// `min(round, last_round)` clamp in `stage_round`; the `K = 32` cases read past the activation.
    #[test]
    fn the_dense_tiled_body_matches_a_host_product_at_every_planned_tiling() {
        let caps = DeviceCaps::rocm_default();
        let (blocks, m, n) = (2usize, 37usize, 41usize);
        for k in [20usize, 17, 32] {
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
                let tiled_shape = TiledShape {
                    blocks,
                    m_per_block: m,
                    out_per_block: n,
                    k,
                    layout: weight.layout(),
                    rows: crate::TiledRows::Dense,
                    terms: StagedTerms::ONE,
                    weight_bits: weight.bits(),
                    activation_bits: 32,
                };
                for tiling in RegisterTiling::candidates(tiled_shape, &caps) {
                    let body = dense_tiled_contraction(
                        "t",
                        weight,
                        tiling.schedule(),
                        super::TiledExtents {
                            k,
                            out_per_block: n,
                        },
                        None,
                        super::TiledRead {
                            precision: poot_graph_ir::OperandPrecision::Exact,
                            activation_stored: None,
                            conversions: caps.rounding_conversions,
                        },
                    )
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

    /// POOT-1448: under a narrowed precision, an activation the planner stored in the fragment dtype is read as stored
    /// (the products read the same values as when the body rounds an f32 activation itself), and an epilogue whose
    /// output is a `Narrow` step stores each sum rounded once: at every planned tiling, over a bf16 `[N, K]` weight,
    /// with a `K` tail and with whole rounds, the narrow body's output is the f32 body's, rounded to nearest even.
    /// Mutation: in `tiled_body` read the activation as f32 whatever its storage (`read_activation(load, None, ..)`);
    /// the interpreter faults on the bf16 buffer read as f32.
    #[test]
    fn a_narrow_stored_activation_and_a_narrowed_store_match_the_f32_body() {
        use crate::{Epilogue, EpilogueInput, EpilogueStep, FusedScalarOp};
        use poot_graph_ir::{FragmentDtype, OperandPrecision};
        use poot_kernel_ir::Ty;
        let caps = DeviceCaps::rocm_default();
        let precision = OperandPrecision::Narrow(FragmentDtype::Bf16);
        let weight = DenseTiledWeight::Nk(BufferStorage::bf16_packed());
        let (m, n) = (37usize, 41usize);
        for k in [20usize, 64] {
            // Activation values that are not bf16 values, so the f32 body rounds them where it stages them.
            let activation: Vec<f32> = values(m * k, 1)
                .iter()
                .enumerate()
                .map(|(i, v)| v + (i % 7) as f32 * 1.0e-3)
                .collect();
            let weights = values(n * k, 2);
            let narrow_store = Epilogue {
                dt: Ty::F32,
                out_shape: vec![m, n],
                leaves: Vec::new(),
                steps: vec![EpilogueStep {
                    op: FusedScalarOp::Narrow(FragmentDtype::Bf16),
                    inputs: vec![EpilogueInput::Head],
                }],
                output: EpilogueInput::Step(0),
            };
            let shape = |activation_bits| TiledShape {
                blocks: 1,
                m_per_block: m,
                out_per_block: n,
                k,
                layout: weight.layout(),
                rows: crate::TiledRows::Dense,
                terms: StagedTerms::of(precision, None, Some(FragmentDtype::Bf16)),
                weight_bits: weight.bits(),
                activation_bits,
            };
            for tiling in RegisterTiling::candidates(shape(16), &caps) {
                let (tile_rows, tile_cols) = tiling.output_tile();
                let tiles = m.div_ceil(tile_rows) * n.div_ceil(tile_cols);
                let meta = [k as u32, n as u32, m as u32, tiles as u32, 0];
                let extents = super::TiledExtents {
                    k,
                    out_per_block: n,
                };
                let interpret = |body: &poot_kernel_ir::Body, activation: Buffer, out: Buffer| {
                    let mut buffers = vec![
                        activation,
                        stored(weight, &weights),
                        Buffer::from_u32s(&meta),
                        out,
                    ];
                    run(body, [tiles as u32, 1, 1], &mut buffers).expect("interpreter");
                    buffers.pop().unwrap()
                };
                let wide = dense_tiled_contraction(
                    "wide",
                    weight,
                    tiling.schedule(),
                    extents,
                    None,
                    super::TiledRead {
                        precision,
                        activation_stored: None,
                        conversions: caps.rounding_conversions,
                    },
                )
                .expect("a planned tiling generates");
                let narrow = dense_tiled_contraction(
                    "narrow",
                    weight,
                    tiling.schedule(),
                    extents,
                    Some(&narrow_store),
                    super::TiledRead {
                        precision,
                        activation_stored: Some(FragmentDtype::Bf16),
                        conversions: caps.rounding_conversions,
                    },
                )
                .expect("a planned tiling generates");
                let want = interpret(
                    &wide,
                    Buffer::from_f32s(&activation),
                    Buffer::from_f32s(&vec![0.0; m * n]),
                )
                .to_f32s()
                .unwrap();
                let got = interpret(
                    &narrow,
                    Buffer::from_halves(Ty::BF16, &activation),
                    Buffer::from_halves(Ty::BF16, &vec![0.0; m * n]),
                )
                .to_halves()
                .unwrap();
                for (i, (got, want)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        got.to_bits(),
                        poot_eval::round_to_fragment(FragmentDtype::Bf16, *want).to_bits(),
                        "K={k} {tiling:?} [{}, {}]: {got} from {want}",
                        i / n,
                        i % n
                    );
                }
            }
        }
    }
}
