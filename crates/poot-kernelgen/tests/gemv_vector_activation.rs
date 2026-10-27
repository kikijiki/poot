//! POOT-1405: the Gemv's vector activation read ([`kg::GemvActivation::Vec4`]) on a dense weight. On the reference
//! interpreter it computes the same bits as the scalar read, including the element-by-element tail a `K` that is not
//! a multiple of the run leaves; on NVPTX the full runs' groups lower to `ld.global.v4` loads, which the scalar read
//! never produces (NVPTX vectorises only naturally aligned accesses, and a scalar read is `align 4`).

use std::process::Command;

use poot_kernel_ir::interp::{Buffer, run};
use poot_kernel_ir::test_support::workgroups_covering;
use poot_kernelgen as kg;
use poot_target::{BufferStorage, DeviceCaps};

/// `K = 36` with 8-element runs over 4 lanes per column: lane 0's second run starts at 32 and has only four
/// elements left, so it takes the element-by-element tail, which reads the activation scalar under either read.
const K: usize = 36;
const N: usize = 7;

fn schedule(activation: kg::GemvActivation) -> kg::Schedule {
    kg::Schedule::Gemv {
        width: 16,
        cols: 4,
        unroll: 8,
        activation,
    }
}

fn generated(activation: kg::GemvActivation) -> kg::Generated {
    let request = kg::KernelRequest::Contraction(kg::ContractionSpec::DenseGemv {
        weight: BufferStorage::f32(),
        schedule: schedule(activation),
        k: K,
        n: N,
        epilogue: None,
    });
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    };
    kg::generate(&request, &DeviceCaps::ptx_default(), &limits)
        .unwrap_or_else(|error| panic!("{activation:?}: {error}"))
}

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

fn interpret(activation: kg::GemvActivation) -> Vec<u32> {
    let generated = generated(activation);
    let mut buffers = [
        Buffer::from_f32s(&fill(3, K)),
        Buffer::from_f32s(&fill(5, N * K)),
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
    run(&generated.body, workgroups, &mut buffers)
        .unwrap_or_else(|error| panic!("{activation:?}: interpreter: {error}"));
    buffers[3]
        .to_f32s()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Mutation: in `gemv_body`, read group `g + 1` instead of `g` for the vector read; the outputs differ (or the
/// interpreter faults past the activation) and this row goes red.
#[test]
fn a_dense_vector_activation_read_matches_the_scalar_read_bit_for_bit() {
    assert_eq!(
        interpret(kg::GemvActivation::Vec4),
        interpret(kg::GemvActivation::Scalar)
    );
}

fn ptx(body: &poot_kernel_ir::Body, tag: &str) -> String {
    let dir = std::env::temp_dir()
        .join("poot-codegen-test")
        .join(format!("gemv-vector-{tag}"));
    std::fs::create_dir_all(&dir).unwrap();
    let out = poot_codegen::artifact_path(&dir, tag, poot_codegen::Target::Nvptx);
    poot_codegen::compile(body, poot_codegen::Target::Nvptx, &out)
        .unwrap_or_else(|error| panic!("{tag}: {error}"));
    std::fs::read_to_string(&out).unwrap()
}

/// Mutation: lower `Rvalue::VectorLoad` on NVPTX with the scalar element alignment (`align 4`) in
/// `emit_rvalue`; llc then splits the group into scalar loads and the `ld.global.v4` count goes to zero.
#[test]
fn nvptx_lowers_the_vector_activation_read_to_v4_loads() {
    if Command::new("llc").arg("--version").output().is_err() {
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
        return;
    }
    let count = |ptx: &str| ptx.matches("ld.global.v4.").count();
    let vector = ptx(&generated(kg::GemvActivation::Vec4).body, "vec4");
    let scalar = ptx(&generated(kg::GemvActivation::Scalar).body, "scalar");
    // One 8-element run: two 4-lane groups.
    assert_eq!(count(&vector), 2, "the vector read's groups\n{vector}");
    assert_eq!(count(&scalar), 0, "the scalar read stays scalar\n{scalar}");
}
