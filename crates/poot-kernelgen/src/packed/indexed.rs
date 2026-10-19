//! The indexed packed contraction's row selection: output row `m` of `out[m, n] = sum over k of x[m, k] *
//! decode(expert ids[m], n, k)` reads the slab of expert `ids[m]` in every source of an expert-stacked table.
//!
//! Row selection is a property of the contraction's index map, not of the decode: the shared Gemv loop nest
//! ([`crate::contraction::gemv_body`]) and the descriptor-driven decode ([`super::decode_plan::DecodePlan`],
//! [`super::planar::PlanarPlan`]) are unchanged, and only the source each decode reads moves. Flat output
//! column `m * N + n` is the "weight row" of the Gemv; this module turns it into the weight row `n` of expert
//! `ids[m]`, each source's slab bit offset (`expert * slab_words * 32`), and the NaN poison of an invalid id.
//!
//! Metadata, in order: `K`, `N`, `E` (the expert count), `M` and `x_groups` (the segmented Tiled schedule's row
//! count and grid X extent, as the dense indexed contraction carries them; the Gemv ignores both), then one word per
//! source, in [`poot_quant::PackedWeight::sources`] order, holding that source's slab stride in `u32` words
//! ([`poot_quant::PackedWeight::slab_words`], the per-expert word padding of the uploaded table).

use poot_kernel_ir::{BinOp, Body, Local, Ty};
use poot_quant::format::Storage;

use super::decode_plan::DecodePlan;
use super::kernel::emit_block_address;
use super::planar::PlanarPlan;
use super::source::Source;
use crate::KernelGenError;
use crate::contraction::{
    Epilogue, IndexedTiledBody, IndexedTiledWeight, META_EXPERTS, META_N, META_PACKED_SLAB_WORDS,
    RowExpert, TileSize, emit_poison, emit_row_expert, indexed_tiled_body,
};
use crate::emit::{Emit, bin, copy, cu, emit_read_meta};
use crate::helpers::{ld, local, slice_dtype, slice_f32};

/// The row selection of an indexed packed Gemv: the `ids` buffer and the `metadata` buffer to read it from.
#[derive(Clone, Copy)]
pub(crate) struct ExpertRows {
    pub(crate) ids: Local,
    pub(crate) metadata: Local,
}

/// What flat output column `row` reads: the weight row inside its expert's slab, the expert's slab in each
/// source, the output row width, and the poison to add to the column's first term.
pub(crate) struct SelectedRow {
    /// The weight row `n` (`row % N`) the decode addresses inside the slab.
    pub(crate) weight_row: Local,
    /// `N`: the weight's `out`, which a planar decode needs as its out extent.
    pub(crate) out_rows: Local,
    /// `0.0` for a valid expert id and NaN otherwise.
    pub(crate) poison: Local,
    /// Bit offset of the selected expert's slab in each source, in source order.
    slab_bits: Vec<Local>,
}

impl SelectedRow {
    /// Source `index` (a buffer of `words`) read relative to the selected expert's slab.
    pub(crate) fn source(&self, index: usize, words: Local) -> Source {
        Source {
            words,
            slab_bit: Some(self.slab_bits[index]),
        }
    }
}

impl ExpertRows {
    /// The selection for flat output column `row` over `sources` source buffers. Emitted at the run read, so
    /// the id is read once per run, at an index every lane of a column computes alike.
    pub(crate) fn select(self, e: &mut Emit, row: Local, sources: usize) -> SelectedRow {
        let n = emit_read_meta(e, self.metadata, META_N);
        let experts = emit_read_meta(e, self.metadata, META_EXPERTS);
        let m = bin(e, Ty::Usize, BinOp::Div, copy(row), copy(n));
        let weight_row = bin(e, Ty::Usize, BinOp::Rem, copy(row), copy(n));
        let RowExpert { expert, valid } = emit_row_expert(e, self.ids, m, experts);
        SelectedRow {
            weight_row,
            out_rows: n,
            poison: emit_poison(e, valid),
            slab_bits: slab_bits(e, self.metadata, expert, sources),
        }
    }
}

/// The bit offset of `expert`'s slab in each of `sources` source buffers: `expert * slab_words * 32`.
fn slab_bits(e: &mut Emit, metadata: Local, expert: Local, sources: usize) -> Vec<Local> {
    (0..sources)
        .map(|source| {
            let words = emit_read_meta(e, metadata, META_PACKED_SLAB_WORDS + source);
            let bits = bin(e, Ty::Usize, BinOp::Mul, copy(words), cu(32));
            bin(e, Ty::Usize, BinOp::Mul, copy(expert), copy(bits))
        })
        .collect()
}

/// The descriptor's decode of an expert-stacked table, by storage: the block layouts address a block in one
/// source ([`DecodePlan`]); a planar layout reads several ([`PlanarPlan`]).
enum Decode {
    Block(DecodePlan),
    Planar(PlanarPlan),
}

impl Decode {
    /// The constant tables the decode reads ([`DecodePlan::constant_tables`]); a planar layout has none.
    fn constant_tables(&self) -> Vec<poot_kernel_ir::ConstantTableDecl> {
        match self {
            Self::Block(plan) => plan.constant_tables(),
            Self::Planar(_) => Vec::new(),
        }
    }
}

/// A stacked packed table as the segmented Tiled body reads it: each lane decodes one weight element of the
/// step's expert, relative to that expert's slab in each source.
struct PackedExperts {
    decode: Decode,
    words: Vec<Local>,
    metadata: Local,
}

impl IndexedTiledWeight for PackedExperts {
    fn step(&self, e: &mut Emit, expert: Local, _extents: (Local, Local)) -> Vec<Local> {
        slab_bits(e, self.metadata, expert, self.words.len())
    }

    fn load(
        &self,
        e: &mut Emit,
        step: &[Local],
        (k_at, row): (Local, Local),
        (k, n): (Local, Local),
    ) -> Local {
        let sources: Vec<Source> = self
            .words
            .iter()
            .zip(step)
            .map(|(&words, &slab_bit)| Source {
                words,
                slab_bit: Some(slab_bit),
            })
            .collect();
        match &self.decode {
            Decode::Block(plan) => {
                let (block, i_in_block) = emit_block_address(e, plan.values(), row, k_at, k);
                let factors = plan.emit_factors(e, sources[0], block, i_in_block);
                plan.emit_value(e, sources[0], block, i_in_block, &factors)
            }
            Decode::Planar(plan) => {
                let factors = plan.emit_factors(e, &sources, row, k_at, k, n);
                plan.emit_value(e, &sources, row, k_at, k, n, &factors)
            }
        }
    }
}

/// The segmented Tiled schedule of an indexed packed contraction (prefill, many rows): params `(activation, ids,
/// sources.., epilogue leaves.., metadata, out)` like the Gemv, the body [`indexed_tiled_body`] shares with the
/// dense indexed contraction, and each lane decoding the packed weight element of the step's expert.
pub(super) fn indexed_tiled(
    name: &str,
    format: poot_quant::format::WeightFormat,
    tile: TileSize,
    epilogue: Option<&Epilogue>,
) -> Result<Body, KernelGenError> {
    let descriptor = format.descriptor();
    let decode = match descriptor.storage {
        Storage::Planar(_) => Decode::Planar(PlanarPlan::of(&descriptor)?),
        _ => Decode::Block(DecodePlan::of(&descriptor)?),
    };
    let sources = match &decode {
        Decode::Block(_) => 1,
        Decode::Planar(plan) => plan.source_count(),
    };
    let leaf_count = epilogue.map_or(0, Epilogue::leaf_count);
    let lanes = match epilogue {
        Some(epilogue) => epilogue.lanes("packed_contraction")?,
        None => Vec::new(),
    };
    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false),
        ld(slice_dtype(Ty::I32, false), false),
    ];
    params.extend((0..sources).map(|_| ld(slice_dtype(Ty::U32, false), false)));
    if let Some(epilogue) = epilogue {
        params.extend(lanes.iter().map(|&lane| epilogue.leaf_param(lane)));
    }
    params.push(ld(slice_dtype(Ty::U32, false), false));
    params.push(ld(slice_f32(true), true));
    let words: Vec<Local> = (0..sources as u32).map(|index| local(3 + index)).collect();
    let leaves: Vec<Local> = (0..leaf_count)
        .map(|j| local(3 + (sources + j) as u32))
        .collect();
    let metadata = local(3 + (sources + leaf_count) as u32);
    let tables = decode.constant_tables();
    let mut body = indexed_tiled_body(
        IndexedTiledBody {
            name,
            tile,
            params,
            x: local(1),
            ids: local(2),
            metadata,
            out: local(4 + (sources + leaf_count) as u32),
            epilogue: epilogue.map(|epilogue| crate::contraction::GemvEpilogue {
                epilogue,
                leaves: &leaves,
                lanes: &lanes,
            }),
        },
        &PackedExperts {
            decode,
            words,
            metadata,
        },
    );
    body.constant_tables = tables;
    Ok(body)
}

#[cfg(test)]
mod tests {
    //! The indexed packed Gemv in the KIR interpreter (GPU-free): every packed layout (block, K-quant and
    //! planar) over a stacked, word-padded table against `PackedPayload::decode_row`.

    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use poot_kernel_ir::interp::{Buffer, run};
    use poot_quant::format::{GroupMap, ScaleEncoding, WeightFormat};
    use poot_quant::{PackedPayload, PackedStack, PackedStackComponentRef, packed_source_words};

    use super::super::fixtures::{GEMV_SCHEDULES, schedule_tag};
    use super::super::kernel::packed_contraction;
    use crate::PackedRows;
    use crate::contraction::{Schedule, TileSize};

    const EXPERTS: usize = 3;

    fn formats() -> Vec<(WeightFormat, usize)> {
        let nz = |n| NonZeroUsize::new(n).unwrap();
        vec![
            (WeightFormat::Q4_0, 64),
            (WeightFormat::Q5_1, 64),
            (WeightFormat::Q8_0, 64),
            (WeightFormat::Q4_K, 256),
            (WeightFormat::Q6_K, 256),
            (WeightFormat::Iq2_Xs, 256),
            (WeightFormat::Iq3_Xxs, 256),
            (WeightFormat::Iq3_S, 256),
            (WeightFormat::Mxfp4, 64),
            (
                WeightFormat::E4m3PerChannel {
                    scale: ScaleEncoding::Bf16,
                },
                64,
            ),
            (
                WeightFormat::E4m3Block128 {
                    scale: ScaleEncoding::Bf16,
                },
                128,
            ),
            (WeightFormat::E2m1Row32, 64),
            (WeightFormat::Awq { group_size: nz(64) }, 128),
            (
                WeightFormat::Gptq {
                    groups: GroupMap::Indexed { groups: nz(4) },
                },
                128,
            ),
        ]
    }

    /// The pieces of one indexed packed launch over `EXPERTS` random experts of `format` at `[out, k]`.
    struct Case {
        stack: Arc<PackedStack>,
        sources: Vec<Vec<u32>>,
    }

    impl Case {
        /// The launch metadata for `rows` activation rows on a grid `x_groups` tiles wide (`0` for a Gemv).
        fn metadata(&self, rows: usize, x_groups: usize) -> Vec<u32> {
            let weight = self.stack.weight();
            let [out, k] = weight.shape();
            [k, out, EXPERTS, rows, x_groups]
                .into_iter()
                .chain(
                    weight
                        .sources()
                        .into_iter()
                        .map(|role| weight.slab_words(role)),
                )
                .map(|word| word as u32)
                .collect()
        }
    }

    fn case(format: WeightFormat, out: usize, k: usize) -> Case {
        case_of(
            (0..EXPERTS)
                .map(|e| {
                    Arc::new(poot_quant::test_support::packed::prescribed_scale_payload(
                        format,
                        [out, k],
                        31 + 7 * e as u64,
                    ))
                })
                .collect(),
        )
    }

    fn case_of(owners: Vec<Arc<PackedPayload>>) -> Case {
        let stack = Arc::new(PackedStack::try_new(owners).unwrap());
        let weight = stack.weight();
        let sources = weight
            .sources()
            .into_iter()
            .map(|role| {
                packed_source_words(
                    &PackedStackComponentRef::new(Arc::clone(&stack), role).slab_bytes(),
                )
            })
            .collect();
        Case { stack, sources }
    }

    /// Every launch shape the indexed rows run: the Gemv shapes and the segmented Tiled schedule at two tile
    /// edges (4: several tiles per row and column group at the fixtures' `N = 5, 6`, so tiles have edge columns;
    /// 8: one tile spans all the fixture rows, so it holds every expert).
    fn schedules() -> Vec<Schedule> {
        let tiled = |edge| Schedule::Tiled {
            tile: TileSize::new(edge).unwrap(),
        };
        GEMV_SCHEDULES
            .into_iter()
            .chain([tiled(4), tiled(8)])
            .collect()
    }

    /// The outputs of `body` for activation `x [m, k]` routed by `ids`; a tiled launch's grid is `fold` tiles
    /// wide when set (folded onto `GroupY`), else the unfolded tile count.
    fn launch(
        case: &Case,
        schedule: Schedule,
        (x, ids): (&[f32], &[i32]),
        name: &str,
        format: WeightFormat,
    ) -> Vec<f32> {
        launch_with(case, schedule, (x, ids), name, format, None, None)
    }

    fn launch_with(
        case: &Case,
        schedule: Schedule,
        (x, ids): (&[f32], &[i32]),
        name: &str,
        format: WeightFormat,
        fold: Option<usize>,
        epilogue: Option<(&crate::contraction::Epilogue, &[f32])>,
    ) -> Vec<f32> {
        let [out, _] = case.stack.weight().shape();
        let m = ids.len();
        let body = packed_contraction(
            name,
            format,
            schedule,
            epilogue.map(|(epilogue, _)| epilogue),
            PackedRows::Indexed,
        )
        .unwrap_or_else(|error| panic!("{format:?}: {error}"));
        let (grid, x_groups) = match schedule {
            Schedule::Gemv { .. } => ([schedule.grid_threads(1, m, out) as u32, 1, 1], 0),
            Schedule::Tiled { tile } => {
                let ts = tile.get() as usize;
                let tiles = m.div_ceil(ts) * out.div_ceil(ts);
                let x_groups = fold.unwrap_or(tiles);
                (
                    [
                        (x_groups * ts * ts) as u32,
                        tiles.div_ceil(x_groups) as u32,
                        1,
                    ],
                    x_groups,
                )
            }
            Schedule::RegisterTiled { .. } => {
                panic!("an indexed contraction is never RegisterTiled")
            }
        };
        let mut buffers = vec![Buffer::from_f32s(x), Buffer::from_i32s(ids)];
        buffers.extend(case.sources.iter().map(|words| Buffer::from_u32s(words)));
        if let Some((_, leaf)) = epilogue {
            buffers.push(Buffer::from_f32s(leaf));
        }
        buffers.push(Buffer::from_u32s(&case.metadata(m, x_groups)));
        buffers.push(Buffer::from_f32s(&vec![f32::MIN; m * out]));
        let last = buffers.len() - 1;
        let wg = body.workgroup_size[0];
        run(&body, [grid[0].div_ceil(wg), grid[1], 1], &mut buffers)
            .unwrap_or_else(|error| panic!("{format:?} {name}: interpreter: {error}"));
        buffers[last].to_f32s().unwrap()
    }

    /// SC-001 (kernel tier, GPU-free): with one-hot activation rows, every output is exactly one decoded
    /// weight, so row `m` routed to expert `ids[m]` equals `decode_row(n, k_m)` of that expert bit for bit
    /// (zero sign canonicalized), for every layout and every Gemv launch shape. Mutation: a slab stride of
    /// zero (`metadata[3 + source]` zeroed); every row reads expert 0 and the rows routed to experts 1 and 2
    /// go red.
    #[test]
    fn one_hot_rows_equal_the_selected_experts_decoded_column() {
        let (out, rows) = (5, 7);
        for (format, k) in formats() {
            let case = case(format, out, k);
            let ids: Vec<i32> = (0..rows).map(|m| ((m * 2 + 1) % EXPERTS) as i32).collect();
            let column = |m: usize| (m * 37 + 11) % k;
            let mut x = vec![0.0f32; rows * k];
            for m in 0..rows {
                x[m * k + column(m)] = 1.0;
            }
            for schedule in schedules() {
                let name = format!("indexed_{format:?}_{}", schedule_tag(schedule));
                let got = launch(&case, schedule, (&x, &ids), &name, format);
                let mut row = vec![0.0f32; k];
                for m in 0..rows {
                    let expert = ids[m] as usize;
                    for n in 0..out {
                        case.stack.decode_row(expert, n, &mut row).unwrap();
                        assert_eq!(
                            (got[m * out + n] + 0.0).to_bits(),
                            (row[column(m)] + 0.0).to_bits(),
                            "{format:?} {schedule:?} row {m} (expert {expert}) col {n}: {} vs {}",
                            got[m * out + n],
                            row[column(m)]
                        );
                    }
                }
            }
        }
    }

    /// A dense random activation makes every output a sum over all of `K`: the launch matches the sequential
    /// reference `sum_k x[m, k] * decode(expert ids[m], n, k)` within f32 tree-reduction tolerance, for every
    /// layout and Gemv shape (a dropped K block or a mis-strided slab moves the sum by far more).
    #[test]
    fn dense_rows_match_the_per_row_expert_product() {
        let (out, rows) = (6, 5);
        for (format, k) in formats() {
            let case = case(format, out, k);
            let ids: Vec<i32> = (0..rows).map(|m| ((m * 2 + 2) % EXPERTS) as i32).collect();
            let x: Vec<f32> = (0..rows * k)
                .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.0625)
                .collect();
            let mut expected = vec![0.0f32; rows * out];
            let mut row = vec![0.0f32; k];
            for m in 0..rows {
                for n in 0..out {
                    case.stack.decode_row(ids[m] as usize, n, &mut row).unwrap();
                    expected[m * out + n] = (0..k).map(|c| x[m * k + c] * row[c]).sum();
                }
            }
            for schedule in schedules() {
                let name = format!("indexed_dense_{format:?}_{}", schedule_tag(schedule));
                let got = launch(&case, schedule, (&x, &ids), &name, format);
                for (i, (got, want)) in got.iter().zip(&expected).enumerate() {
                    // An inf weight meets a zero activation as NaN in both.
                    assert!(
                        (got.is_nan() && want.is_nan())
                            || (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                        "{format:?} {schedule:?} output {i}: {got} vs {want}"
                    );
                }
            }
        }
    }

    /// MXFP4 scale bytes at the ends of the admitted range cycle over the blocks of every expert and a one-hot row
    /// is compared with the oracle's sum `x . decode_row(n)` bit for bit. With `0xfe` (2^127) in the table, the
    /// oracle's weights overflow to inf and every row's `0 * inf` is NaN: the kernel must propagate it, not
    /// flush or clamp it. With `0xfc` (2^125) and `0x00` (2^-127, an f32 subnormal) every weight stays finite
    /// and the sums are exact large and tiny values. A scale byte of `0xff` (NaN) is not an admitted payload
    /// (`PackedPayload::try_new` refuses it; `poot-quant`'s `an_all_zero_block_is_valid...` test pins that), so
    /// no kernel can be handed one. Mutation: clamp the E8M0 decode with `min(bits, 0xf0)`; both cases go red.
    #[test]
    fn extreme_e8m0_scales_propagate_like_the_oracle() {
        let (out, k, rows) = (4, 128, 16);
        let format = WeightFormat::Mxfp4;
        for (scales, expect_non_finite) in
            [([0x00u8, 0xfe, 0x7f], true), ([0x00, 0xfc, 0x7f], false)]
        {
            let case = case_of(
                (0..EXPERTS)
                    .map(|e| {
                        Arc::new(poot_quant::test_support::packed::e8m0_scaled_payload(
                            format,
                            [out, k],
                            31 + 7 * e as u64,
                            |block| scales[(block + e) % scales.len()],
                        ))
                    })
                    .collect(),
            );
            let ids: Vec<i32> = (0..rows).map(|m| (m % EXPERTS) as i32).collect();
            // Row `m` is hot in 32-wide scale block `m % 4`, at a column that moves with `m`.
            let mut x = vec![0.0f32; rows * k];
            for m in 0..rows {
                x[m * k + 32 * (m % 4) + (m * 5) % 32] = 1.0;
            }
            for schedule in schedules() {
                let name = format!("indexed_e8m0_extreme_{}", schedule_tag(schedule));
                let got = launch(&case, schedule, (&x, &ids), &name, format);
                let mut row = vec![0.0f32; k];
                let mut non_finite = 0;
                for m in 0..rows {
                    for n in 0..out {
                        case.stack.decode_row(ids[m] as usize, n, &mut row).unwrap();
                        let want: f32 = (0..k).map(|c| x[m * k + c] * row[c]).sum();
                        non_finite += usize::from(!want.is_finite());
                        let value = got[m * out + n];
                        assert!(
                            (value.is_nan() && want.is_nan())
                                || (value + 0.0).to_bits() == (want + 0.0).to_bits(),
                            "{scales:x?} {schedule:?} row {m} col {n}: {value} vs oracle {want}"
                        );
                    }
                }
                assert_eq!(
                    non_finite > 0,
                    expect_non_finite,
                    "{scales:x?}: fixture reach"
                );
            }
        }
    }

    /// An expert id outside `0..E` (negative, at or past `E`, or `i32::MIN`) makes its whole output row NaN and leaves
    /// every other row untouched (R-632-4), as for the dense indexed contraction; no read leaves the table.
    #[test]
    fn an_invalid_expert_id_makes_only_its_row_nan() {
        let (out, k) = (4, 64);
        let format = WeightFormat::Q8_0;
        let case = case(format, out, k);
        let ids = [1, EXPERTS as i32, -1, 7 * EXPERTS as i32, i32::MIN, 2];
        let x: Vec<f32> = (0..ids.len() * k).map(|i| 0.25 + (i % 5) as f32).collect();
        for schedule in schedules() {
            let name = format!("indexed_invalid_ids_{}", schedule_tag(schedule));
            let got = launch(&case, schedule, (&x, &ids), &name, format);
            let mut row = vec![0.0f32; k];
            for (m, &id) in ids.iter().enumerate() {
                let valid = (0..EXPERTS as i32).contains(&id);
                for n in 0..out {
                    let value = got[m * out + n];
                    if valid {
                        case.stack.decode_row(id as usize, n, &mut row).unwrap();
                        let want: f32 = (0..k).map(|c| x[m * k + c] * row[c]).sum();
                        assert!(
                            (value - want).abs() <= 1e-3 * want.abs().max(1.0),
                            "{schedule:?} row {m}: {value} vs {want}"
                        );
                    } else {
                        assert!(
                            value.is_nan(),
                            "{schedule:?} row {m} (id {id}) col {n}: {value}"
                        );
                    }
                }
            }
        }
    }

    /// The routed gate weighting fuses into the indexed Gemv as an epilogue: `out[m, n] = (sum_k x[m, k] *
    /// decode(expert ids[m], n, k)) * gate[m]` with the gate a `[M, 1]` leaf between the table's sources and
    /// the metadata, applied once to each completed sum (an epilogue applied per lane would scale every
    /// partial sum and miss the reference for every launch shape with more than one lane per column).
    #[test]
    fn the_gate_weighting_epilogue_scales_each_rows_completed_sum() {
        use crate::contraction::{
            Epilogue, EpilogueInput as In, EpilogueLeaf, EpilogueStep as Step,
        };
        use crate::fused::FusedScalarOp as Op;
        let (out, rows, k) = (6, 5, 64);
        let format = WeightFormat::Q8_0;
        let case = case(format, out, k);
        let ids: Vec<i32> = (0..rows).map(|m| ((m * 2 + 1) % EXPERTS) as i32).collect();
        let x: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 5 % 19) as f32 - 9.0) * 0.0625)
            .collect();
        let gate: Vec<f32> = (0..rows).map(|m| 0.25 + 0.5 * m as f32).collect();
        let epilogue = Epilogue {
            dt: poot_kernel_ir::Ty::F32,
            out_shape: vec![rows, out],
            leaves: vec![EpilogueLeaf {
                shape: vec![rows, 1],
                storage: None,
            }],
            steps: vec![Step {
                op: Op::Binary(poot_kernel_ir::BinOp::Mul),
                inputs: vec![In::Head, In::Leaf(0)],
            }],
            output: In::Step(0),
        };
        for schedule in schedules() {
            let name = format!("indexed_epilogue_{}", schedule_tag(schedule));
            let got = launch_with(
                &case,
                schedule,
                (&x, &ids),
                &name,
                format,
                None,
                Some((&epilogue, &gate)),
            );
            let mut row = vec![0.0f32; k];
            for m in 0..rows {
                for n in 0..out {
                    case.stack.decode_row(ids[m] as usize, n, &mut row).unwrap();
                    let dot: f32 = (0..k).map(|c| x[m * k + c] * row[c]).sum();
                    let want = dot * gate[m];
                    let value = got[m * out + n];
                    assert!(
                        (value - want).abs() <= 1e-3 * want.abs().max(1.0),
                        "{schedule:?} row {m} col {n}: {value} vs {want}"
                    );
                }
            }
        }
    }

    /// The segmented walk: a tile whose rows select a subset of the experts with a gap (here experts 0 and 2 of
    /// three, in an order that is neither sorted nor grouped) reads each row's own expert slab at every step.
    /// With edge-4 tiles the 8 rows are two tiles, and the first holds ids `{2, 0, 2, 0}`: the step after expert 0
    /// must land on expert 2, not on the next integer. Mutation: in `indexed_tiled_body` take the step's slab
    /// from `lo` instead of `next`; the rows routed to expert 2 read expert 1 and go red.
    #[test]
    fn a_tile_whose_rows_skip_an_expert_reads_each_rows_own_slab() {
        let (out, k) = (5, 64);
        for format in [WeightFormat::Q8_0, WeightFormat::Q4_K] {
            let k = if format == WeightFormat::Q4_K { 256 } else { k };
            let case = case(format, out, k);
            let ids = [2, 0, 2, 0, 0, 2, 2, 0];
            let rows = ids.len();
            let x: Vec<f32> = (0..rows * k)
                .map(|i| ((i * 11 % 29) as f32 - 14.0) * 0.03125)
                .collect();
            let schedule = Schedule::Tiled {
                tile: TileSize::new(4).unwrap(),
            };
            let got = launch(
                &case,
                schedule,
                (&x, &ids),
                "indexed_skipped_expert",
                format,
            );
            let mut row = vec![0.0f32; k];
            for m in 0..rows {
                for n in 0..out {
                    case.stack.decode_row(ids[m] as usize, n, &mut row).unwrap();
                    let want: f32 = (0..k).map(|c| x[m * k + c] * row[c]).sum();
                    let value = got[m * out + n];
                    assert!(
                        (value - want).abs() <= 1e-3 * want.abs().max(1.0),
                        "{format:?} row {m} (expert {}) col {n}: {value} vs {want}",
                        ids[m]
                    );
                }
            }
        }
    }

    /// A tiled grid folded onto `GroupY` (here 3 tiles wide over 4 tiles) computes what the unfolded grid does,
    /// bit for bit: `g = gy * x_groups + gx` recovers the tile, and the padding workgroups of the last fold row
    /// store nothing. Mutation: read `g` from `GroupX` alone; the tile past the first fold row is never computed
    /// (its outputs stay `f32::MIN`) and this row goes red.
    #[test]
    fn a_folded_tiled_grid_equals_the_unfolded_one() {
        let (out, rows, k) = (6, 8, 64);
        let format = WeightFormat::Q8_0;
        let case = case(format, out, k);
        let ids: Vec<i32> = (0..rows).map(|m| ((m * 2 + 1) % EXPERTS) as i32).collect();
        let x: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.0625)
            .collect();
        let schedule = Schedule::Tiled {
            tile: TileSize::new(4).unwrap(),
        };
        let run_with = |fold| {
            launch_with(
                &case,
                schedule,
                (&x, &ids),
                "indexed_fold",
                format,
                fold,
                None,
            )
        };
        let (unfolded, folded) = (run_with(None), run_with(Some(3)));
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&folded), bits(&unfolded));
        assert!(
            unfolded.iter().all(|&v| v != f32::MIN),
            "every output is stored"
        );
    }
}
