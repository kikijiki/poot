//! POOT-1108: an `I32ToF32` step of a float pointwise region on the kernel-IR reference interpreter (no GPU).
//! Every row builds its body through `generate`, the seam the planner uses, and runs it on the launch that call
//! returns. The expectations are hand-written words: the conversion rounds to nearest even, an unsigned compare
//! of I32 storage bits is an exact integer step, and the cast turns its 0/1 into the float the arithmetic after it
//! reads.

use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen as kg;
use poot_target::DeviceCaps;

fn limits() -> kg::BodyLimits {
    kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    }
}

fn leaf(numel: usize) -> kg::ViewOperand {
    kg::ViewOperand {
        shape: vec![numel],
        layout: kg::Layout::contiguous(&[numel]),
    }
}

/// Generate a flat float region over `numel` elements and run it on `buffers` (leaves, then the output).
fn run_region(numel: usize, kernel: kg::FusedKernel, buffers: &mut [Buffer]) -> Vec<u32> {
    let spec = kg::PointwiseSpec {
        out_shape: vec![numel],
        numel,
        leaves: (0..kernel.n_leaves()).map(|_| leaf(numel)).collect(),
        kernel,
        form: kg::PointwiseForm::Float {
            fold: kg::Fold {
                numel,
                two_d: false,
                width: 256,
            },
        },
    };
    let generated = kg::generate(
        &kg::KernelRequest::Pointwise(spec),
        &DeviceCaps::wgpu_rdna3_igpu(),
        &limits(),
    )
    .expect("the request generates");
    let workgroup = generated.body.workgroup_size;
    let workgroups =
        std::array::from_fn(|axis| generated.launch.grid[axis].div_ceil(workgroup[axis]));
    run(&generated.body, workgroups, buffers).expect("the interpreter runs the region");
    buffers
        .last()
        .map(|out| out.to_f32s().expect("an f32 output"))
        .expect("an output buffer")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// A cast of an I32 leaf converts to nearest even: 2^24 + 1 is not an f32 and rounds to 2^24, -1 and 0 are
/// exact, and `i32::MIN` is -2^31.
#[test]
fn a_cast_step_converts_an_i32_leaf_to_the_nearest_f32() {
    let words = [16_777_217, -1, 0, i32::MIN, 3, 16_777_219];
    let kernel = kg::FusedKernel {
        leaf_tys: vec![kg::LeafTy::I32],
        steps: vec![kg::FusedStep {
            op: kg::FusedScalarOp::I32ToF32,
            inputs: vec![kg::FusedInput::Leaf(0)],
        }],
        output: kg::FusedInput::Step(0),
    };
    let mut buffers = [
        Buffer::from_i32s(&words),
        // A value no conversion produces, so an unwritten lane is visible.
        Buffer::from_f32s(&[f32::from_bits(0xDEAD_BEEF); 6]),
    ];
    let got = run_region(words.len(), kernel, &mut buffers);
    let want: Vec<u32> = [
        16_777_216.0f32,
        -1.0,
        0.0,
        -2_147_483_648.0,
        3.0,
        16_777_220.0,
    ]
    .iter()
    .map(|v| v.to_bits())
    .collect();
    assert_eq!(got, want);
}

/// An exact-integer chain and the float arithmetic that reads it are one region: `GeU` over I32 storage bits
/// (so -1 is the largest word, not a negative number), the cast of its 0/1, then a float multiply.
#[test]
fn an_exact_i32_chain_then_a_cast_then_float_arithmetic_is_one_region() {
    let a = [-1, 0, 5, 7, 2, i32::MIN];
    let b = [1, 0, 5, 9, 3, 1];
    let c = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let kernel = kg::FusedKernel {
        leaf_tys: vec![kg::LeafTy::I32, kg::LeafTy::I32, kg::LeafTy::F32],
        steps: vec![
            kg::FusedStep {
                op: kg::FusedScalarOp::GeU,
                inputs: vec![kg::FusedInput::Leaf(0), kg::FusedInput::Leaf(1)],
            },
            kg::FusedStep {
                op: kg::FusedScalarOp::I32ToF32,
                inputs: vec![kg::FusedInput::Step(0)],
            },
            kg::FusedStep {
                op: kg::FusedScalarOp::Binary(poot_kernel_ir::BinOp::Mul),
                inputs: vec![kg::FusedInput::Step(1), kg::FusedInput::Leaf(2)],
            },
        ],
        output: kg::FusedInput::Step(2),
    };
    let mut buffers = [
        Buffer::from_i32s(&a),
        Buffer::from_i32s(&b),
        Buffer::from_f32s(&c),
        Buffer::from_f32s(&[f32::from_bits(0xDEAD_BEEF); 6]),
    ];
    let got = run_region(a.len(), kernel, &mut buffers);
    // GeU(a, b): -1 >= 1 and i32::MIN (2^31) >= 1 hold unsigned; 0 >= 0, 5 >= 5 hold; 7 >= 9 and 2 >= 3 do not.
    let want: Vec<u32> = [10.0f32, 20.0, 30.0, 0.0, 0.0, 60.0]
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(got, want);
}
