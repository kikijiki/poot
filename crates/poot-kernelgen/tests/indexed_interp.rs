//! The indexed contraction's bodies on the kernel-IR reference interpreter, without a GPU. The interpreter faults on
//! any out-of-bounds read, so a body that reads the weight at an expert past the table fails here by name rather
//! than by a device-dependent value: this is where "an invalid expert id yields a NaN row and never an out-of-range
//! read" is observable.

use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen as kg;
use poot_target::{BufferStorage, DeviceCaps};

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
    let w = weights();
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
    let values = stored(layout);
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

/// The generated request for `m` rows under `schedule`, its kernel run on the interpreter, the `[m, N]` output.
fn run_indexed(
    schedule: kg::Schedule,
    m: usize,
    x_groups: Option<u32>,
    layout: kg::WeightLayout,
    weight: BufferStorage,
) -> Vec<f32> {
    let caps = DeviceCaps::wgpu_rdna3_igpu();
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    };
    let (grid, x_groups) = match schedule {
        kg::Schedule::Gemv { .. } => ([schedule.grid_threads(1, m, N) as u32, 1, 1], 0),
        kg::Schedule::Tiled { tile } => {
            let ts = tile.get() as usize;
            let tiles = (m.div_ceil(ts) * N.div_ceil(ts)) as u32;
            // A folded grid when asked: `x_groups` tiles along X and as many rows of them as cover the rest.
            let x_groups = x_groups.unwrap_or(tiles);
            let y_groups = tiles.div_ceil(x_groups);
            ([x_groups * (ts * ts) as u32, y_groups, 1], x_groups)
        }
        kg::Schedule::RegisterTiled { .. } => {
            panic!("an indexed contraction is never RegisterTiled")
        }
    };
    let request =
        kg::KernelRequest::Contraction(kg::ContractionSpec::Indexed(kg::IndexedContraction {
            schedule,
            layout,
            weight,
            epilogue: None,
            grid,
            meta: vec![K as u32, N as u32, EXPERTS as u32, m as u32, x_groups],
        }));
    let generated = kg::generate(&request, &caps, &limits).unwrap();
    let wg = generated.body.workgroup_size[0];
    let mut buffers = [
        Buffer::from_f32s(&fill(1, m * K)),
        weight_buffer(weight, layout),
        Buffer::from_i32s(&ids(m)),
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

/// Every refusal id yields a NaN row and no read leaves the table, on the Gemv and on the segmented Tiled schedule
/// (the latter also over a folded grid, which the device rows never reach), over an f32 weight and over a BF16 one
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
    };
    let tiled = kg::Schedule::Tiled {
        tile: kg::TileSize::new(4).unwrap(),
    };
    for weight in [BufferStorage::f32(), BufferStorage::bf16_packed()] {
        for layout in [kg::WeightLayout::Kn, kg::WeightLayout::Nk] {
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
            // 6 row-tiles x 3 column tiles = 18 workgroups folded onto a 5-wide X grid: the body rebuilds
            // `gy * x_groups + gx`.
            let m = 21;
            assert_rows(
                &label("tiled folded"),
                &run_indexed(tiled, m, Some(5), layout, weight),
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
            },
            layout: kg::WeightLayout::Kn,
            weight: BufferStorage::f16_packed(),
            epilogue: None,
            grid: [16, 1, 1],
            meta: vec![K as u32, N as u32, EXPERTS as u32, 1, 0],
        }));
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    };
    let error = kg::generate(&request, &DeviceCaps::wgpu_rdna3_igpu(), &limits)
        .expect_err("an F16 indexed weight has no body");
    assert!(error.to_string().contains("an indexed weight"), "{error}");
}
