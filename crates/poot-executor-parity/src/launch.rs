//! Launch-shape fixtures: the generated tiled GEMM with a remainder tile, bf16 matmuls that run on
//! the tensor cores, and one dispatch over more workgroups than a byte of workgroup id can name.

use poot_graph_ir::{Builder, Slot, TensorType};
use poot_tensor::{DType, HostTensor};

use poot_eval::test_support::StepFixture;

use crate::{Fixture, fill, store_from_tensors, store_with};

/// A `[m, k] x [k, n]` f32 matmul: `x` is an activation slot (two steps with different data), `w` a
/// constant.
fn matmul_fixture(name: &'static str, m: usize, k: usize, n: usize) -> Fixture {
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![m, k]))
        .unwrap();
    let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let out = b.matmul(x, w).unwrap();
    let graph = b.finish(out).unwrap();
    let key = graph.meta(x.id()).slot_key().unwrap().clone();
    let steps = (0..2u64)
        .map(|step| {
            vec![StepFixture {
                key: key.clone(),
                tensor: HostTensor::f32(vec![m, k], fill(m * k, 0x71_ED + step)),
            }]
        })
        .collect();
    let store = store_with(&graph, |_, _| {});
    Fixture {
        name,
        graph,
        store,
        steps,
    }
}

/// The generated tiled GEMM (`TiledRegion`) at the gemma4-MoE router shape that once hung gfx1151
/// (`M = 32, K = 2816, N = 128`, `[L, H] x [H, E]`), and the same with a remainder tile on both
/// output axes (`M = 33`, `N = 130`: not a multiple of the tile edge), where the last tile in each
/// direction is partial.
pub fn tiled_region_fixtures() -> Vec<Fixture> {
    vec![
        matmul_fixture("tiled_region_router", 32, 2816, 128),
        matmul_fixture("tiled_region_router_remainder_tile", 33, 2816, 130),
    ]
}

/// `n` small integers in `[-range, range]`.
fn small_integers(n: usize, seed: u64, range: i32) -> Vec<i32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 33) as i32).rem_euclid(2 * range + 1) - range
        })
        .collect()
}

/// A mixed-precision `[s, s] x [s, s]` matmul: bf16 operands, f32 accumulate and result, the one shape
/// the planner runs on the RDNA3 WMMA tensor cores (every dimension a multiple of 16, no batch
/// dimension; a bf16 result is planned as the serial kernel instead). No tracer emits it any more (the
/// `to_mixed_bf16` pass that rewrote f32 matmuls into it is deleted), so the fixture states it directly:
/// the matmul is the f32-accumulate form, f32 over bf16 operands. `x` is an f32 step slot (the contract binds F32 and I32 slots) cast to bf16
/// in the program, `w` a bf16 constant held as its stored words. Both are small integers (`|x| <= 1`,
/// `|w| <= 2`), so every product and sum is exact in the bf16 operands and the f32 accumulator, and the
/// device's result equals the oracle's bit for bit whatever its summation order.
fn wmma_fixture(name: &'static str, s: usize) -> Fixture {
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![s, s]))
        .unwrap();
    let w = b
        .constant("w", TensorType::new(vec![s, s], DType::BF16))
        .unwrap();
    let out = b
        .matmul_f32_accumulate(b.cast(x, DType::BF16).unwrap(), w)
        .unwrap();
    let graph = b.finish(out).unwrap();
    let key = graph.meta(x.id()).slot_key().unwrap().clone();
    let steps = (0..2u64)
        .map(|step| {
            let values = small_integers(s * s, 0xB0 + step, 1);
            vec![StepFixture {
                key: key.clone(),
                tensor: HostTensor::f32(vec![s, s], values.iter().map(|&v| v as f32).collect()),
            }]
        })
        .collect();
    // bf16 of a small integer is the high half of its f32 bits: exact.
    let words = small_integers(s * s, 0xFE, 2)
        .iter()
        .map(|&v| (f32::to_bits(v as f32) >> 16) as u16)
        .collect();
    let store = store_from_tensors(vec![("w", HostTensor::bf16(vec![s, s], words))]);
    Fixture {
        name,
        graph,
        store,
        steps,
    }
}

/// WMMA bf16 at one tile (`16 x 16 x 16`) and at a `4 x 4` grid of tiles with a four-step K loop
/// (`64 x 64 x 64`).
pub fn wmma_bf16_fixtures() -> Vec<Fixture> {
    vec![
        wmma_fixture("wmma_bf16_16x16x16", 16),
        wmma_fixture("wmma_bf16_64x64x64", 64),
    ]
}

/// Rows, K and columns of [`wmma_batched_fixtures`]: two batch elements of `32 x 48 x 32`, so 2 x 2 x 2 output
/// tiles and three `16`-wide K blocks per tile, with a batched right operand.
const WMMA_BATCH: usize = 2;
const WMMA_M: usize = 32;
const WMMA_K: usize = 48;
const WMMA_N: usize = 32;

/// A batched mixed-precision matmul `[2, 32, 48] x [2, 48, 32]` over `operand` (BF16 or F16) operands with f32
/// accumulate and result: every output tile accumulates three K blocks, and each batch element reads its own
/// right operand. `x` is an f32 step slot cast to `operand` in the program, `w` a constant of `operand`.
///
/// The values are small positive integers chosen so that every K block contributes a distinct nonzero
/// amount to every output: `x[b, i, k] = 1 + (i + b + step) % 2` and
/// `w[b, k, j] = (k / 16 + 1) * (1 + j % 3) + b`, so block `t` of output `(b, i, j)` adds
/// `16 * x * ((t + 1) * (1 + j % 3) + b)`, at least 16. A lost block, a block accumulated twice or a block read
/// from the wrong batch element moves every output element by at least 16 against a parity tolerance orders
/// of magnitude smaller; every product and sum is exact in the operand type and the f32 accumulator.
fn wmma_batched_fixture(name: &'static str, operand: DType) -> Fixture {
    let b = Builder::for_tests();
    let x = b
        .slot_named(
            Slot::Activation,
            "x",
            TensorType::f32(vec![WMMA_BATCH, WMMA_M, WMMA_K]),
        )
        .unwrap();
    let w = b
        .constant(
            "w",
            TensorType::new(vec![WMMA_BATCH, WMMA_K, WMMA_N], operand),
        )
        .unwrap();
    let out = b
        .matmul_f32_accumulate(b.cast(x, operand).unwrap(), w)
        .unwrap();
    let graph = b.finish(out).unwrap();
    let key = graph.meta(x.id()).slot_key().unwrap().clone();
    let steps = (0..2usize)
        .map(|step| {
            let values = (0..WMMA_BATCH * WMMA_M * WMMA_K).map(|flat| {
                let (batch, row) = (flat / (WMMA_M * WMMA_K), flat / WMMA_K % WMMA_M);
                (1 + (row + batch + step) % 2) as f32
            });
            vec![StepFixture {
                key: key.clone(),
                tensor: HostTensor::f32(vec![WMMA_BATCH, WMMA_M, WMMA_K], values.collect()),
            }]
        })
        .collect();
    let weights = (0..WMMA_BATCH * WMMA_K * WMMA_N).map(|flat| {
        let (batch, k, col) = (
            flat / (WMMA_K * WMMA_N),
            flat / WMMA_N % WMMA_K,
            flat % WMMA_N,
        );
        ((k / 16 + 1) * (1 + col % 3) + batch) as f32
    });
    let tensor = match operand {
        DType::BF16 => HostTensor::bf16(
            vec![WMMA_BATCH, WMMA_K, WMMA_N],
            // bf16 of a small integer is the high half of its f32 bits: exact.
            weights.map(|v| (v.to_bits() >> 16) as u16).collect(),
        ),
        DType::F16 => HostTensor::f16(
            vec![WMMA_BATCH, WMMA_K, WMMA_N],
            weights.map(crate::packed::f16_bits).collect(),
        ),
        other => panic!("a tensor-core fixture takes BF16 or F16 operands, not {other:?}"),
    };
    Fixture {
        name,
        graph,
        store: store_from_tensors(vec![("w", tensor)]),
        steps,
    }
}

/// The batched multi-block tensor-core matmul over bf16 operands (RDNA3 WMMA, NVPTX WMMA).
pub fn wmma_bf16_batched_fixture() -> Fixture {
    wmma_batched_fixture("wmma_bf16_batched_k48", DType::BF16)
}

/// The batched multi-block tensor-core matmul over f16 operands (SPIR-V cooperative matrix).
pub fn wmma_f16_batched_fixture() -> Fixture {
    wmma_batched_fixture("wmma_f16_batched_k48", DType::F16)
}

/// Elements of the wide-grid fixture: past `256 * 256`, so even a 256-lane workgroup needs more than
/// 256 workgroups, and not a multiple of any workgroup size, so the last one is partial.
pub const WIDE_GRID_ELEMENTS: usize = 70_001;

/// One elementwise region (`x * y + x`) over [`WIDE_GRID_ELEMENTS`] elements: a single dispatch whose
/// grid needs `workgroup_id.x > 255`.
pub fn wide_grid_fixture() -> Fixture {
    let n = WIDE_GRID_ELEMENTS;
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![n]))
        .unwrap();
    let y = b
        .slot_named(Slot::Activation, "y", TensorType::f32(vec![n]))
        .unwrap();
    let out = b
        .binary(
            poot_graph_ir::BinOp::Add,
            b.binary(poot_graph_ir::BinOp::Mul, x, y).unwrap(),
            x,
        )
        .unwrap();
    let graph = b.finish(out).unwrap();
    let (kx, ky) = (
        graph.meta(x.id()).slot_key().unwrap().clone(),
        graph.meta(y.id()).slot_key().unwrap().clone(),
    );
    let steps = (0..2u64)
        .map(|step| {
            vec![
                StepFixture {
                    key: kx.clone(),
                    tensor: HostTensor::f32(vec![n], fill(n, 0x91 + step)),
                },
                StepFixture {
                    key: ky.clone(),
                    tensor: HostTensor::f32(vec![n], fill(n, 0x92 + step)),
                },
            ]
        })
        .collect();
    Fixture {
        name: "wide_grid_elementwise",
        graph,
        store: store_from_tensors(Vec::new()),
        steps,
    }
}
