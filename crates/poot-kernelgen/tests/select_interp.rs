//! POOT-1033: the generated `Select` kernel on the kernel-IR reference interpreter (no GPU), so `just test`
//! guards the emitter's mask and not only the device lanes. Every row builds its body through `generate`, the
//! seam the planner uses, and runs it on the launch that call returns.
//!
//! The expectations are hand-written bit words: a condition of 2 and -1 selects `if_true`, 0 selects
//! `if_false`, a NaN or an infinity in the unselected branch stays out, and a NaN payload, an infinity and
//! `-0.0` in the selected one come out bit for bit.

use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen as kg;
use poot_target::DeviceCaps;

const NAN_PAYLOAD: u32 = 0x7FC0_1234;
const SIGNALING_NAN: u32 = 0x7F80_0001;
const POSITIVE_INFINITY: u32 = 0x7F80_0000;
const NEGATIVE_INFINITY: u32 = 0xFF80_0000;
const NEGATIVE_ZERO: u32 = 0x8000_0000;

const CONDITION: [i32; 6] = [1, 0, 2, -1, 0, 1];
const IF_TRUE: [u32; 6] = [
    NAN_PAYLOAD,
    1.0f32.to_bits(),
    POSITIVE_INFINITY,
    NEGATIVE_ZERO,
    5.0f32.to_bits(),
    NEGATIVE_INFINITY,
];
const IF_FALSE: [u32; 6] = [
    3.0f32.to_bits(),
    SIGNALING_NAN,
    7.0f32.to_bits(),
    NAN_PAYLOAD,
    NEGATIVE_ZERO,
    NAN_PAYLOAD,
];
const EXPECTED: [u32; 6] = [
    NAN_PAYLOAD,
    SIGNALING_NAN,
    POSITIVE_INFINITY,
    NEGATIVE_ZERO,
    NEGATIVE_ZERO,
    NEGATIVE_INFINITY,
];

fn limits() -> kg::BodyLimits {
    kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    }
}

/// `Select(cond, if_true, if_false)` over `numel` elements, lane `i` reading table entry `i % 6`, generated for
/// `caps` in `form` and interpreted over the launch `generate` returns.
fn run_select(
    numel: usize,
    caps: &DeviceCaps,
    float: bool,
    fold: kg::Fold,
) -> (Vec<u32>, kg::Launch) {
    let leaf = || kg::ViewOperand {
        shape: vec![numel],
        layout: kg::Layout::contiguous(&[numel]),
    };
    let branch = if float {
        kg::LeafTy::F32
    } else {
        kg::LeafTy::I32
    };
    let spec = kg::PointwiseSpec {
        out_shape: vec![numel],
        numel,
        leaves: vec![leaf(), leaf(), leaf()],
        kernel: kg::FusedKernel {
            leaf_tys: vec![kg::LeafTy::I32, branch, branch],
            steps: vec![kg::FusedStep {
                op: kg::FusedScalarOp::Select,
                inputs: vec![
                    kg::FusedInput::Leaf(0),
                    kg::FusedInput::Leaf(1),
                    kg::FusedInput::Leaf(2),
                ],
            }],
            output: kg::FusedInput::Step(0),
        },
        form: if float {
            kg::PointwiseForm::Float { fold }
        } else {
            kg::PointwiseForm::ExactI32 { fold }
        },
    };
    let generated = kg::generate(&kg::KernelRequest::Pointwise(spec), caps, &limits())
        .expect("a select request generates");
    // The planner's `finalize` bakes the fold's workgroup width into a folded body; do the same here.
    let mut body = generated.body;
    if fold.two_d {
        body.workgroup_size = [fold.width as u32, 1, 1];
    }
    let workgroup = body.workgroup_size;
    let workgroups =
        std::array::from_fn(|axis| generated.launch.grid[axis].div_ceil(workgroup[axis]));
    let cond: Vec<i32> = (0..numel).map(|i| CONDITION[i % 6]).collect();
    let words = |table: &[u32; 6]| -> Vec<u32> { (0..numel).map(|i| table[i % 6]).collect() };
    let as_buffer = |bits: Vec<u32>| {
        if float {
            Buffer::from_f32s(&bits.iter().map(|&w| f32::from_bits(w)).collect::<Vec<_>>())
        } else {
            Buffer::from_i32s(&bits.iter().map(|&w| w as i32).collect::<Vec<_>>())
        }
    };
    let mut buffers = [
        Buffer::from_i32s(&cond),
        as_buffer(words(&IF_TRUE)),
        as_buffer(words(&IF_FALSE)),
        // Every output word starts as a value no select produces, so an unwritten lane is visible.
        as_buffer(vec![0xDEAD_BEEF; numel]),
    ];
    run(&body, workgroups, &mut buffers).expect("the interpreter runs the select");
    let out = if float {
        buffers[3]
            .to_f32s()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    } else {
        buffers[3]
            .to_i32s()
            .unwrap()
            .iter()
            .map(|&w| w as u32)
            .collect()
    };
    (out, generated.launch)
}

fn flat(numel: usize) -> kg::Fold {
    kg::Fold {
        numel,
        two_d: false,
        width: 256,
    }
}

fn expected(numel: usize) -> Vec<u32> {
    (0..numel).map(|i| EXPECTED[i % 6]).collect()
}

/// An F32 select over poisoned branches equals the hand-written words: both the NaN/Inf stay-out lanes and the
/// payload passthrough lanes, and a condition of 2 and -1 picks `if_true`. Mutation: change the emitter's
/// `Ne 0` to `Eq 1` in `emit_select`; the cond 2 and -1 lanes select `if_false` and this row goes red.
#[test]
fn float_select_equals_the_hand_written_words() {
    let (got, _) = run_select(6, &DeviceCaps::wgpu_rdna3_igpu(), true, flat(6));
    assert_eq!(got, EXPECTED, "got {got:#010x?}");
}

/// The I32 form takes the same mask: a nonzero word of any value (2, -1) picks `if_true`, and the words pass
/// through even where `if_true - if_false` would wrap.
#[test]
fn i32_select_takes_the_true_branch_for_any_nonzero_word() {
    let (got, _) = run_select(6, &DeviceCaps::wgpu_rdna3_igpu(), false, flat(6));
    // The i32 form moves the float table's words as plain words.
    assert_eq!(got, EXPECTED, "got {got:#010x?}");
}

/// A float select on a device whose X grid cap is below the workgroup count folds onto Y: the launch
/// is 2-D, every lane is written exactly once, and each holds its hand-written word. 700 elements in 256-wide
/// workgroups are 3 workgroups, over a cap of 2.
#[test]
fn a_folded_float_select_covers_every_element_once() {
    let numel = 700;
    let mut narrow = DeviceCaps::wgpu_rdna3_igpu();
    narrow.max_grid[0] = 2;
    let fold = kg::Fold {
        numel,
        two_d: true,
        width: 256,
    };
    let (got, launch) = run_select(numel, &narrow, true, fold);
    assert_eq!(
        launch.grid,
        [2 * 256, 2, 1],
        "3 workgroups fold onto 2 across and 2 down"
    );
    let wrong = got
        .iter()
        .zip(expected(numel))
        .filter(|(g, w)| **g != *w)
        .count();
    assert_eq!(wrong, 0, "{wrong} of {numel} lanes are unwritten or wrong");
}
