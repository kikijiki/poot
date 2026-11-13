//! The indexed contraction's bodies on the kernel-IR reference interpreter, without a GPU. The interpreter faults on
//! any out-of-bounds read, so a body that reads the weight at an expert past the table fails here by name rather
//! than by a device-dependent value: this is where "an invalid expert id yields a NaN row and never an out-of-range
//! read" is observable.

use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen as kg;
use poot_target::BufferStorage;

const EXPERTS: usize = 4;
const K: usize = 10;
const N: usize = 9;

/// Ids covering every refusal (`4`, `-1`, a huge id, `i32::MIN`, `i32::MAX`) around valid and repeated ones.
fn ids(rows: usize) -> Vec<i32> {
    let cycle = [
        2,
        0,
        4,
        3,
        -1,
        2,
        i32::MIN,
        1,
        i32::MAX,
        1_000_000_000,
        0,
        -7,
        3,
        1,
    ];
    (0..rows).map(|r| cycle[r % cycle.len()]).collect()
}

fn valid(id: i32) -> bool {
    (0..EXPERTS as i32).contains(&id)
}

fn fill(seed: usize, count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 7 + seed * 13) % 7) as f32 - 3.0)
        .collect()
}

/// The logical weight `W[expert, k, n]` as a flat `[E, K, N]` stack: multiples of 0.25 within a few units, which a
/// BF16 holds exactly and whose products and sums are exact in f32 in any order.
fn weights() -> Vec<f32> {
    (0..EXPERTS * K * N)
        .map(|i| ((i * 7 + 2 * 13) % 11) as f32 * 0.25 - 1.25)
        .collect()
}

/// `weights()` in the order `layout` stores each expert slab.
fn stored(layout: kg::WeightLayout) -> Vec<f32> {
    arranged(layout, weights())
}

/// The logical weights `w`, a flat `[E, K, N]` stack, in the order `layout` stores each expert slab.
fn arranged(layout: kg::WeightLayout, w: Vec<f32>) -> Vec<f32> {
    match layout {
        kg::WeightLayout::Kn => w,
        kg::WeightLayout::Nk => (0..EXPERTS * K * N)
            .map(|at| {
                let (expert, in_slab) = (at / (K * N), at % (K * N));
                let (n, k) = (in_slab / K, in_slab % K);
                w[expert * K * N + k * N + n]
            })
            .collect(),
    }
}

/// The weight buffer in `storage`: f32 words, or BF16 packed two per `u32` word (element `i` in word `i / 2` at bit
/// `(i % 2) * 16`), the bytes a BF16 checkpoint uploads.
fn weight_buffer(storage: BufferStorage, layout: kg::WeightLayout) -> Buffer {
    stored_buffer(storage, stored(layout))
}

/// The weight buffer holding `values`, already in a layout's stored order, in `storage`.
fn stored_buffer(storage: BufferStorage, values: Vec<f32>) -> Buffer {
    if storage == BufferStorage::f32() {
        return Buffer::from_f32s(&values);
    }
    let bf16 = |value: f32| {
        let bits = value.to_bits();
        assert_eq!(bits & 0xffff, 0, "{value} is not exactly a BF16");
        bits >> 16
    };
    let words: Vec<u32> = values
        .chunks(2)
        .map(|pair| bf16(pair[0]) | pair.get(1).map_or(0, |&value| bf16(value) << 16))
        .collect();
    Buffer::from_u32s(&words)
}

/// The block-aligned grouping of `ids` in tiles of `block` ([`poot_graph_ir::ExpertGrouping`]), built bucket by
/// bucket on the host: what an `ExpertGroup` computes.
fn grouping(ids: &[i32], block: usize) -> Vec<i32> {
    let layout = poot_graph_ir::ExpertGrouping {
        rows: ids.len(),
        experts: EXPERTS,
        block,
    };
    let bucket = |id: i32| if valid(id) { id as usize } else { EXPERTS };
    let (mut order, mut tiles) = (Vec::new(), Vec::new());
    for b in 0..=EXPERTS {
        let rows: Vec<i32> = (0..ids.len())
            .filter(|&r| bucket(ids[r]) == b)
            .map(|r| r as i32)
            .collect();
        for chunk in rows.chunks(block) {
            order.extend_from_slice(chunk);
            order.resize(order.len() + block - chunk.len(), ids.len() as i32);
            tiles.push(b as i32);
        }
    }
    order.resize(layout.positions(), ids.len() as i32);
    tiles.resize(layout.tiles(), EXPERTS as i32);
    order.extend(tiles);
    order
}

/// The generated request for `m` rows under `schedule`, its kernel run on the interpreter, the `[m, N]` output.
fn run_indexed(
    schedule: kg::Schedule,
    m: usize,
    x_groups: Option<u32>,
    layout: kg::WeightLayout,
    weight: BufferStorage,
) -> Vec<f32> {
    run_indexed_over(
        schedule,
        (m, x_groups),
        (layout, weight),
        (fill(1, m * K), weight_buffer(weight, layout)),
    )
}

/// [`run_indexed`] over the given activation rows and weight buffer.
fn run_indexed_over(
    schedule: kg::Schedule,
    (m, x_groups): (usize, Option<u32>),
    (layout, weight): (kg::WeightLayout, BufferStorage),
    (activation, weights): (Vec<f32>, Buffer),
) -> Vec<f32> {
    let caps = poot_test_util::device_caps::wgpu_rdna3_igpu();
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    };
    // The Gemv reads the ids; a tiled schedule reads their grouping in tiles of its output rows, whose positions
    // are its grid rows.
    let (grid, x_groups, rows, experts) = match schedule {
        kg::Schedule::Gemv { .. } => ([schedule.grid_threads(1, m, N) as u32, 1, 1], 0, m, ids(m)),
        kg::Schedule::RegisterTiled { .. } | kg::Schedule::FragmentTiled { .. } => {
            let (tile_rows, tile_cols) = schedule.output_tile().unwrap();
            let groups = grouping(&ids(m), tile_rows);
            let positions = poot_graph_ir::ExpertGrouping {
                rows: m,
                experts: EXPERTS,
                block: tile_rows,
            }
            .positions();
            let tiles = (positions.div_ceil(tile_rows) * N.div_ceil(tile_cols)) as u32;
            let lanes = kg::RegisterTiling::of(schedule).unwrap().workgroup_lanes() as u32;
            // A folded grid when asked: `x_groups` tiles along X and as many rows of them as cover the rest.
            let x_groups = x_groups.unwrap_or(tiles);
            let y_groups = tiles.div_ceil(x_groups);
            ([x_groups * lanes, y_groups, 1], x_groups, positions, groups)
        }
    };
    let request =
        kg::KernelRequest::Contraction(kg::ContractionSpec::Indexed(kg::IndexedContraction {
            schedule,
            layout,
            weight,
            epilogue: None,
            grid,
            meta: vec![K as u32, N as u32, rows as u32, x_groups, 0, EXPERTS as u32],
            precision: poot_graph_ir::OperandPrecision::Exact,
        }));
    let generated = kg::generate(&request, &caps, &limits).unwrap();
    let wg = generated.body.workgroup_size[0];
    let mut buffers = [
        Buffer::from_f32s(&activation),
        weights,
        Buffer::from_i32s(&experts),
        Buffer::from_u32s(generated.launch.meta.as_ref().unwrap()),
        Buffer::from_f32s(&vec![f32::MIN; m * N]),
    ];
    run(
        &generated.body,
        [generated.launch.grid[0] / wg, generated.launch.grid[1], 1],
        &mut buffers,
    )
    .unwrap_or_else(|e| panic!("{schedule:?} over {m} rows: {e}"));
    buffers[4].to_f32s().unwrap()
}

fn assert_rows(label: &str, got: &[f32], m: usize) {
    let (x, w, ids) = (fill(1, m * K), weights(), ids(m));
    for (row, &id) in ids.iter().enumerate() {
        for col in 0..N {
            let value = got[row * N + col];
            if valid(id) {
                let expert = id as usize;
                let want = (0..K).fold(0.0f32, |acc, k| {
                    acc + x[row * K + k] * w[expert * K * N + k * N + col]
                });
                assert_eq!(
                    value.to_bits(),
                    want.to_bits(),
                    "{label} row {row} col {col}"
                );
            } else {
                assert!(
                    value.is_nan(),
                    "{label} row {row} (id {id}) col {col}: {value}"
                );
            }
        }
    }
}

/// Every refusal id yields a NaN row and no read leaves the table, on the Gemv (per-row ids) and on the register-tiled
/// schedule over the rows' grouping (also over a folded grid, which the device rows never reach), over an f32 weight and over a BF16 one
/// held packed two per word (the weight buffer then has half the words, so the clamp bounds a word read too), in
/// both expert layouts, and every valid row is bit-exact with the sequential f32 sum over the widened weights.
///
/// Mutations: in `emit_row_expert` drop the clamp of the cast id (`Min(raw, E - 1)`): the interpreter faults on the
/// weight read of the `4` row ("index ... is out of bounds for length ..."). In `WeightLane::decode_low_half` read
/// the BF16 bits without the `Shl 16` (or swap the halves in `emit_elem`): every BF16 row mismatches its sum.
#[test]
fn an_invalid_expert_id_is_a_nan_row_and_never_an_out_of_range_read() {
    let gemv = kg::Schedule::Gemv {
        width: 16,
        cols: 2,
        unroll: 2,
        activation: poot_kernelgen::GemvActivation::Scalar,
        rows: poot_kernelgen::GemvRows::ONE,
        depth: poot_kernelgen::LoadDepth::ONE,
    };
    for weight in [BufferStorage::f32(), BufferStorage::bf16_packed()] {
        for layout in [kg::WeightLayout::Kn, kg::WeightLayout::Nk] {
            // The cost model's tiling for this small shape: one 16-row tile per 16 rows.
            let tiled = kg::RegisterTiling::choose(
                kg::TiledShape {
                    blocks: 1,
                    m_per_block: 21,
                    out_per_block: N,
                    k: K,
                    layout,
                    rows: kg::TiledRows::Grouped {
                        groups: EXPERTS + 1,
                        block: 16,
                    },
                    terms: poot_kernelgen::StagedTerms::ONE,
                    weight: kg::StagedWeight::Dense {
                        bits: kg::DenseTiledWeight::Nk(weight).bits(),
                    },
                    activation_bits: 32,
                },
                &poot_test_util::device_caps::wgpu_rdna3_igpu(),
            )
            .schedule();
            assert_eq!(tiled.output_tile(), Some((16, 32)), "{tiled:?}");
            let label = |schedule: &str| format!("{schedule} {weight:?} {layout:?}");
            let m = 14;
            assert_rows(
                &label("gemv"),
                &run_indexed(gemv, m, None, layout, weight),
                m,
            );
            assert_rows(
                &label("tiled"),
                &run_indexed(tiled, m, None, layout, weight),
                m,
            );
            // The grouping's row tiles folded onto a 1-wide X grid: the body rebuilds `gy * x_groups + gx`.
            let m = 21;
            assert_rows(
                &label("tiled folded"),
                &run_indexed(tiled, m, Some(1), layout, weight),
                m,
            );
        }
    }
}

/// An indexed weight is f32 or BF16 packed: an F16-packed request is refused with a typed `Unsupported` naming the
/// indexed weight, not generated.
///
/// Mutation: drop the `F16Packed` arm of `indexed_contraction`; the request generates and this row goes red.
#[test]
fn an_f16_packed_indexed_weight_is_refused_by_name() {
    let request =
        kg::KernelRequest::Contraction(kg::ContractionSpec::Indexed(kg::IndexedContraction {
            schedule: kg::Schedule::Gemv {
                width: 16,
                cols: 2,
                unroll: 2,
                activation: poot_kernelgen::GemvActivation::Scalar,
                rows: poot_kernelgen::GemvRows::ONE,
                depth: poot_kernelgen::LoadDepth::ONE,
            },
            layout: kg::WeightLayout::Kn,
            weight: BufferStorage::f16_packed(),
            epilogue: None,
            grid: [16, 1, 1],
            meta: vec![K as u32, N as u32, 1, 0, 0, EXPERTS as u32],
            precision: poot_graph_ir::OperandPrecision::Exact,
        }));
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    };
    let error = kg::generate(
        &request,
        &poot_test_util::device_caps::wgpu_rdna3_igpu(),
        &limits,
    )
    .expect_err("an F16 indexed weight has no body");
    assert!(error.to_string().contains("an indexed weight"), "{error}");
}

/// POOT-1560 SC-e on the indexed Gemv: a lane that accumulates `rows` adjacent output columns of one row's expert
/// slab (N = 9, so 3 divides it) and keeps `depth` trips of loads in flight computes every output, the NaN rows of
/// the refused ids included, bit for bit as at one row and one trip, over both expert layouts and both weight storages.
/// The inputs have full mantissas (the BF16 weights ones a BF16 holds), so a sum in another order changes bits; the
/// fixture of the invalid-id row sums exactly in any order and would not show it.
/// Mutation: in `gemv_body` run the products of rows after the first in reverse order of the run (the `MUTATION`
/// line); every row count above one goes red.
#[test]
fn indexed_gemv_rows_and_depth_do_not_change_a_sum() {
    // Two lanes per column read 2 values each: a lane sums three runs of two products in sequence (K = 10), so the
    // order of its products is visible in the bits.
    let gemv = |rows, depth| kg::Schedule::Gemv {
        width: 4,
        cols: 2,
        unroll: 2,
        activation: poot_kernelgen::GemvActivation::Scalar,
        rows: poot_kernelgen::GemvRows::new(rows).unwrap(),
        depth: poot_kernelgen::LoadDepth::new(depth).unwrap(),
    };
    let hard = |seed: u64, count: usize, bf16: bool| -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let value = ((state >> 40) as f32 / (1u64 << 24) as f32) * 0.8 - 0.4;
                if bf16 {
                    f32::from_bits(value.to_bits() & 0xffff_0000)
                } else {
                    value
                }
            })
            .collect()
    };
    let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let m = 14;
    for weight in [BufferStorage::f32(), BufferStorage::bf16_packed()] {
        for layout in [kg::WeightLayout::Kn, kg::WeightLayout::Nk] {
            let bf16 = weight == BufferStorage::bf16_packed();
            let run = |schedule| {
                run_indexed_over(
                    schedule,
                    (m, None),
                    (layout, weight),
                    (
                        hard(3, m * K, false),
                        stored_buffer(weight, arranged(layout, hard(5, EXPERTS * K * N, bf16))),
                    ),
                )
            };
            let want = bits(run(gemv(1, 1)));
            for (rows, depth) in [(3, 1), (1, 2), (3, 2), (1, 3), (3, 3)] {
                assert_eq!(
                    bits(run(gemv(rows, depth))),
                    want,
                    "{weight:?} {layout:?} rows={rows} depth={depth}"
                );
            }
        }
    }
}
