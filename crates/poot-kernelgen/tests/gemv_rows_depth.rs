//! POOT-1560 SC-e on the dense Gemv: a lane that accumulates `rows` adjacent output columns and keeps `depth` trips
//! of weight loads in flight computes every column's sum bit for bit as at one row and one trip, at the same lanes
//! per column and run, over the f32, f16-packed and bf16-packed weight lanes, with a `K` the run divides, one it
//! leaves a tail of one element on, and an odd one that reads a packed weight element by element.

use poot_kernel_ir::interp::{Buffer, run};
use poot_kernel_ir::test_support::workgroups_covering;
use poot_kernelgen as kg;
use poot_target::{BufferStorage, DeviceCaps};

/// Output columns: a multiple of every row count, over 4 column groups, so the last workgroup of 4 rows per lane
/// leaves columns past the output.
const N: usize = 12;

/// Values with full mantissas, so a product meeting the wrong activation, or a sum in another order, changes bits.
fn fill(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 0.8 - 0.4
        })
        .collect()
}

/// The weight buffer for `n * k` values in `storage`: f32, or halves two to a word.
fn weights(storage: BufferStorage, count: usize) -> Buffer {
    let values = fill(5, count);
    if storage == BufferStorage::f32() {
        return Buffer::from_f32s(&values);
    }
    let half = |value: f32| {
        if storage == BufferStorage::bf16_packed() {
            u32::from(poot_quant::scalar::f32_to_bf16(value))
        } else {
            u32::from(poot_quant::scalar::f32_to_f16(value))
        }
    };
    let words: Vec<u32> = values
        .chunks(2)
        .map(|pair| half(pair[0]) | pair.get(1).map_or(0, |&value| half(value) << 16))
        .collect();
    Buffer::from_u32s(&words)
}

fn interpret(storage: BufferStorage, k: usize, rows: u32, depth: u32) -> Vec<u32> {
    let request = kg::KernelRequest::Contraction(kg::ContractionSpec::DenseGemv {
        weight: storage,
        schedule: kg::Schedule::Gemv {
            width: 16,
            cols: 2,
            unroll: 4,
            activation: kg::GemvActivation::Scalar,
            rows: kg::GemvRows::new(rows).unwrap(),
            depth: kg::LoadDepth::new(depth).unwrap(),
        },
        k,
        n: N,
        epilogue: None,
    });
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    };
    let generated = kg::generate(&request, &DeviceCaps::ptx_default(), &limits)
        .unwrap_or_else(|error| panic!("{storage:?} k={k} rows={rows} depth={depth}: {error}"));
    let mut buffers = [
        Buffer::from_f32s(&fill(3, k)),
        weights(storage, N * k),
        Buffer::from_u32s(
            &generated
                .launch
                .meta
                .clone()
                .expect("the Gemv takes [K, N]"),
        ),
        Buffer::from_f32s(&[0.0; N]),
    ];
    let workgroups = workgroups_covering(&generated.body, generated.launch.grid[0] as usize);
    run(&generated.body, workgroups, &mut buffers).unwrap_or_else(|error| {
        panic!("{storage:?} k={k} rows={rows} depth={depth}: interpreter: {error}")
    });
    buffers[3]
        .to_f32s()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Mutation: in `gemv_body` run the products of rows after the first in reverse order of the run (the `MUTATION`
/// line); every row count above one goes red. Mutation: drop the `fits` guard of the prefetch block; the last
/// row's prefetch reads past the weight and the interpreter faults.
#[test]
fn dense_rows_and_depth_do_not_change_a_rows_sum() {
    for storage in [
        BufferStorage::f32(),
        BufferStorage::f16_packed(),
        BufferStorage::bf16_packed(),
    ] {
        // Eight lanes per column read 4 values each, 32 per trip: 200 is six full trips and a run of four, 64 is two
        // trips, 37 a trip, a run and a tail of one element, 33 a trip and a tail of one with no full run left.
        for k in [200, 64, 37, 33] {
            let want = interpret(storage, k, 1, 1);
            for rows in 1..=4 {
                for depth in 1..=3 {
                    assert_eq!(
                        interpret(storage, k, rows, depth),
                        want,
                        "{storage:?} k={k} rows={rows} depth={depth}"
                    );
                }
            }
        }
    }
}
