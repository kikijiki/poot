//! Launch-shape fixtures: the generated tiled GEMM with a remainder tile, bf16 matmuls that run on
//! the tensor cores, and one dispatch over more workgroups than a byte of workgroup id can name.

use poot_graph_ir::{Builder, Slot, TensorType};
use poot_tensor::HostTensor;

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

/// The register-tiled GEMM over an f32 `[K, N]` weight at the gemma4-MoE router shape that once hung gfx1151
/// (`M = 32, K = 2816, N = 128`, `[L, H] x [H, E]`), and the same with a remainder tile on both
/// output axes (`M = 33`, `N = 130`: not a multiple of the tile edge), where the last tile in each
/// direction is partial.
pub fn tiled_gemm_fixtures() -> Vec<Fixture> {
    vec![
        matmul_fixture("tiled_gemm_router", 32, 2816, 128),
        matmul_fixture("tiled_gemm_router_remainder_tile", 33, 2816, 130),
    ]
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
