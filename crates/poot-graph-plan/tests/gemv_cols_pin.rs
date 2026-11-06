//! POOT-1490: a typed columns pin on the compile options makes the planner launch the dense decode Gemv at one named
//! columns-per-workgroup, so the CUDA calibration sweep can time a weight row split over different lane counts. Every
//! row drives the production `compile` and reads the schedule of the kernel request the plan holds.

use poot_graph_ir::{Builder, Graph, MatmulPrecision, OpKind, TensorType};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Program, Target, compile};
use poot_kernelgen::{ContractionSpec, KernelRequest, Schedule};
use poot_target::Backend;
use poot_tensor::DType;

/// A decode step of the bank's dense projection: one row against a `[N, K]` bf16 checkpoint weight.
const K: usize = 1024;
const N: usize = 4096;

fn wgpu() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
    }
}

fn decode_bf16() -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![N, K], DType::BF16))
        .unwrap();
    let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    b.finish(out).unwrap()
}

fn compiled(pin: Option<u32>) -> Program {
    let options = CompileOptions {
        gemv_cols_pin: pin,
        ..CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(MatmulPrecision::Exact)
    };
    compile(&decode_bf16(), &wgpu(), &options).expect("the decode projection plans")
}

/// The Gemv schedule of the dense contraction's kernel request, or `None` when the plan is not a Gemv.
fn gemv_of(program: &Program) -> Option<Schedule> {
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::DenseContraction { .. }))
        .expect("the dense contraction is planned");
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseGemv {
            schedule,
            ..
        })) => Some(*schedule),
        _ => None,
    }
}

fn cols_of(schedule: Schedule) -> u32 {
    match schedule {
        Schedule::Gemv { cols, .. } => cols,
        other => panic!("not a Gemv: {other:?}"),
    }
}

/// Each columns count the workgroup width divides, pinned, is the columns the Gemv launches with; the device's own
/// choice is one of them and the others differ from it, so a pin that was ignored would launch the same columns for all.
/// Mutation: in `dense_decode_gemv_in`, ignore `cols_pin`; every pinned compile but the planner's own choice goes red
/// on `pinned compile launched other columns`.
#[test]
fn a_pinned_column_count_is_the_one_launched() {
    let default = cols_of(gemv_of(&compiled(None)).expect("the unpinned plan is a Gemv"));
    let pins = [1, 2, 4, 8, 16];
    assert!(
        pins.contains(&default),
        "the device's choice {default} is a pin"
    );
    for cols in pins {
        let planned = cols_of(gemv_of(&compiled(Some(cols))).expect("a pinned plan is a Gemv"));
        assert_eq!(planned, cols, "pinned compile launched other columns");
    }
}

/// Production planning is unchanged when no pin is set: pinning the columns the device chose plans the same kernel.
/// Mutation: in `dense_decode_gemv_in`, rebuild the pinned launch with `unroll: 1`; the pinned plan differs from the
/// unpinned one.
#[test]
fn pinning_the_chosen_columns_changes_nothing() {
    let unpinned = compiled(None);
    let chosen = cols_of(gemv_of(&unpinned).expect("a Gemv"));
    let text = |p: &Program| {
        p.planned()
            .map(|(eqn, _)| format!("{:?}", p.kernel_choice(eqn)))
            .collect::<Vec<_>>()
    };
    assert_eq!(text(&compiled(Some(chosen))), text(&unpinned));
}

/// Columns the workgroup width does not divide launch no Gemv: the planner declines it and plans the contraction as it
/// would without one, never other columns in the pin's place.
/// Mutation: drop the divisibility check in `dense_decode_gemv_in`; the pinned plan is a Gemv of 3 columns.
#[test]
fn columns_the_width_does_not_divide_launch_no_gemv() {
    assert_eq!(gemv_of(&compiled(Some(3))), None);
    assert_eq!(gemv_of(&compiled(Some(0))), None);
}
