//! Flash-attention decode on the GPU (card 044 / spec 055, Card 557).
//!
//! Traces one decode attention (GQA + masked softmax for one query token against a fixed-capacity KV) with
//! `ops::attention_masked` and runs it through the executor contract. `compile` fuses the chain into one
//! `FlashAttentionDecode`, and the planner lowers it to the synthesized flash decode
//! (`kg::flash_region_decode`, online softmax with the running output in LDS), so it runs on wgpu/Vulkan as
//! well as NVIDIA.
//!
//! The fixture is hand-checkable: 1 head, head_dim 2, two KV positions, scale 1.0, no masking.
//!   q = [1, 0];  k0 = [1, 0], k1 = [0, 1]  ->  scores = [q.k0, q.k1] = [1, 0]
//!   softmax([1, 0]) = [e/(e+1), 1/(e+1)] = [0.7311, 0.2689]
//!   v0 = [10, 20], v1 = [30, 40]  ->  out = 0.7311*v0 + 0.2689*v1 = [15.38, 25.38]
//!
//! Run with: `cargo run -p poot-gpu --example flash_attention` (inside `nix develop`; needs a Vulkan
//! adapter - it skips cleanly without one).

use poot_executor::Device as _;
use poot_executor_parity::ConstFixture;
use poot_gpu::device::WgpuDevice;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::types::TensorType;
use poot_tensor::HostTensor;

fn const_row(name: &'static str, shape: Vec<usize>, values: &[f32]) -> ConstFixture {
    ConstFixture {
        name,
        tensor: HostTensor::f32(shape, values.to_vec()),
    }
}

fn main() {
    let (hq, hkv, cap, d) = (1usize, 1usize, 2usize, 2usize);
    let n_rep = hq / hkv;
    let scale = 1.0f32;

    // trace one decode attention: q[1,Hq,1,D], k/v[1,Hkv,cap,D], mask[..,cap] -> [1,Hq,1,D].
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
    let y = poot_graph_ir::ops::attention_masked(&b, q, k, v, n_rep, scale, mask);
    let g = b.finish(y);

    let device = match WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); this example needs a Vulkan adapter. Skipping.");
            return;
        }
    };
    let target = device.target();
    let mut engine = poot_executor::Engine::new(device);

    let consts = [
        const_row("q", vec![1, hq, 1, d], &[1.0, 0.0]),
        const_row("k", vec![1, hkv, cap, d], &[1.0, 0.0, 0.0, 1.0]),
        const_row("v", vec![1, hkv, cap, d], &[10.0, 20.0, 30.0, 40.0]),
        const_row("mask", vec![1, 1, 1, cap], &[0.0, 0.0]),
    ];
    let output = poot_executor_parity::run_once(&mut engine, target, &g, &consts, &[])
        .expect("run the flash-attention kernel through the executor contract");
    let out = output.as_f32().expect("the attention output is F32");

    // hand reference: w = softmax([1, 0]); out = w0*v0 + w1*v1.
    let (w0, w1) = {
        let (e1, e0) = (1.0f32.exp(), 1.0f32);
        (e1 / (e1 + e0), e0 / (e1 + e0))
    };
    let want = [w0 * 10.0 + w1 * 30.0, w0 * 20.0 + w1 * 40.0];

    println!(
        "flash-attention output = [{:.4}, {:.4}] (expected [{:.4}, {:.4}])",
        out[0], out[1], want[0], want[1]
    );
    for (got, exp) in out.iter().zip(&want) {
        assert!(
            (got - exp).abs() < 1e-3,
            "the flash kernel must match the hand-computed attention output"
        );
    }
    println!(
        "this is the same kernel poot dispatches for decode attention: `compile` fuses the traced \
         chain and the planner chooses the synthesized flash decode."
    );
}
