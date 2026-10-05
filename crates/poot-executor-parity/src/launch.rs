//! Launch-shape fixtures: the generated tiled GEMM with a remainder tile, bf16 matmuls that run on
//! the tensor cores, and one dispatch over more workgroups than a byte of workgroup id can name.

use poot_graph_ir::{Builder, Slot, TensorType};
use poot_graph_plan::FusionPolicy;
use poot_tensor::{DType, HostTensor};

use poot_test_util::StepFixture;

use crate::{Fixture, fill, store_from_tensors, store_with};

/// A `[m, k] x [k, n]` f32 matmul: `x` is an activation slot (two steps with different data), `w` a
/// constant.
fn matmul_fixture(name: &'static str, m: usize, k: usize, n: usize) -> Fixture {
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::f32(vec![k, n]));
    let out = b.matmul(x, w);
    let graph = b.finish(out);
    let key = graph.meta(x.id).slot_key().unwrap().clone();
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
        fusion: FusionPolicy::Full,
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
/// the matmul's stored output type is f32 over bf16 operands, which `Graph` validation admits as an
/// f32-accumulate widening. `x` is an f32 step slot (the contract binds F32 and I32 slots) cast to bf16
/// in the program, `w` a bf16 constant held as its stored words. Both are small integers (`|x| <= 1`,
/// `|w| <= 2`), so every product and sum is exact in the bf16 operands and the f32 accumulator, and the
/// device's result equals the oracle's bit for bit whatever its summation order.
fn wmma_fixture(name: &'static str, s: usize) -> Fixture {
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![s, s]));
    let w = b.constant("w", TensorType::new(vec![s, s], DType::BF16));
    let out = b.matmul(b.cast(x, DType::BF16), w);
    let mut graph = b.finish(out);
    graph.values[out.id].aval.dtype = DType::F32;
    let key = graph.meta(x.id).slot_key().unwrap().clone();
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
        fusion: FusionPolicy::Full,
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

/// Elements of the wide-grid fixture: past `256 * 256`, so even a 256-lane workgroup needs more than
/// 256 workgroups, and not a multiple of any workgroup size, so the last one is partial.
pub const WIDE_GRID_ELEMENTS: usize = 70_001;

/// One elementwise region (`x * y + x`) over [`WIDE_GRID_ELEMENTS`] elements: a single dispatch whose
/// grid needs `workgroup_id.x > 255`.
pub fn wide_grid_fixture() -> Fixture {
    let n = WIDE_GRID_ELEMENTS;
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![n]));
    let y = b.slot_named(Slot::Activation, "y", TensorType::f32(vec![n]));
    let out = b.binary(
        poot_graph_ir::BinOp::Add,
        b.binary(poot_graph_ir::BinOp::Mul, x, y),
        x,
    );
    let graph = b.finish(out);
    let (kx, ky) = (
        graph.meta(x.id).slot_key().unwrap().clone(),
        graph.meta(y.id).slot_key().unwrap().clone(),
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
        fusion: FusionPolicy::Full,
    }
}
