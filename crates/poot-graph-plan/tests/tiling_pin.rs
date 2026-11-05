//! POOT-1494: a typed tiling pin on the compile options makes the planner run one named register tiling, so the
//! calibration sweep can time every candidate. Every row drives the production `compile` and reads the schedule of
//! the kernel request the plan holds.

use poot_graph_ir::{Builder, Graph, MatmulPrecision, OpKind, TensorType};
use poot_graph_plan::{
    Capability, CompileError, CompileLimits, CompileOptions, KernelChoice, PlanError, Program,
    Target, compile,
};
use poot_kernelgen::{
    ContractionSpec, KernelRequest, RegisterTiling, Schedule, StagedTerms, TiledRows, TiledShape,
    WeightLayout,
};
use poot_target::Backend;
use poot_tensor::DType;

/// The long prefill projection of the bank (`prefill-dense-bf16-nk`): 2048 rows by a 3072 x 1024 weight.
const M: usize = 2048;
const K: usize = 1024;
const N: usize = 3072;

fn wgpu() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
    }
}

/// The same device with a 64-invocation workgroup limit: it runs no 256-lane tiling.
fn narrow_workgroups() -> Target {
    let Target::Gpu { backend, mut caps } = wgpu() else {
        unreachable!("wgpu() is a GPU target")
    };
    caps.max_workgroup_invocations = 64;
    caps.max_workgroup_size[0] = 64;
    Target::Gpu { backend, caps }
}

/// `matmul(x, transpose(w))` with a bf16 `[N, K]` checkpoint weight: the dense fold turns it into one
/// `DenseContraction` over the weight as stored.
fn dense_bf16() -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![M, K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![N, K], DType::BF16))
        .unwrap();
    let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    b.finish(out).unwrap()
}

/// The shape the planner tiles `dense_bf16` for.
fn shape() -> TiledShape {
    TiledShape {
        blocks: 1,
        m_per_block: M,
        out_per_block: N,
        k: K,
        layout: WeightLayout::Nk,
        rows: TiledRows::Dense,
        terms: StagedTerms::ONE,
        weight_bits: 16,
        activation_bits: 32,
    }
}

fn candidates(target: &Target) -> Vec<RegisterTiling> {
    let Target::Gpu { caps, .. } = target else {
        unreachable!("a GPU target")
    };
    RegisterTiling::candidates(shape(), caps).collect()
}

fn options(pin: Option<RegisterTiling>) -> CompileOptions {
    CompileOptions {
        tiling_pin: pin,
        ..CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(MatmulPrecision::Exact)
    }
}

fn compiled(target: &Target, pin: Option<RegisterTiling>) -> Result<Program, CompileError> {
    compile(&dense_bf16(), target, &options(pin))
}

/// The schedule of the dense contraction's kernel request.
fn schedule_of(program: &Program) -> Schedule {
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::DenseContraction { .. }))
        .expect("the dense contraction is planned");
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled(spec))) => {
            spec.schedule
        }
        other => panic!("expected a generated tiled contraction, got {other:?}"),
    }
}

/// Every kernel choice of the plan, in order, as its `Debug` text: the whole request, grid and metadata included.
fn plan_text(program: &Program) -> Vec<String> {
    program
        .planned()
        .map(|(eqn, _)| format!("{:?}", program.kernel_choice(eqn)))
        .collect()
}

/// Each candidate tiling the device runs, pinned, plans exactly that tiling; the candidates are several and the
/// planner's own choice is one of them, so a pin that was ignored would plan the same schedule for all of them.
/// Mutation: in `register_tiled_launch`, ignore `site.tiling_pin()`; every pinned compile but the planner's own
/// choice goes red on `pinned compile planned another schedule`.
#[test]
fn a_pinned_tiling_is_the_tiling_planned() {
    let target = wgpu();
    let default = schedule_of(&compiled(&target, None).unwrap());
    let candidates = candidates(&target);
    assert!(candidates.len() > 1, "{} candidates", candidates.len());
    assert!(
        candidates.iter().any(|t| t.schedule() == default),
        "the planner's choice {default:?} is a candidate"
    );
    for tiling in &candidates {
        let planned = schedule_of(&compiled(&target, Some(*tiling)).unwrap());
        assert_eq!(
            planned,
            tiling.schedule(),
            "pinned compile planned another schedule"
        );
    }
    assert!(
        candidates.iter().any(|t| t.schedule() != default),
        "no candidate differs from the default, so the pin could not be told from no pin"
    );
}

/// Production planning is unchanged when no pin is set: the plan holds the tiling the cost model chooses
/// (`RegisterTiling::choose`, asked here directly), and pinning that same tiling changes nothing: the whole plan
/// (every request, grid and metadata) is the same text.
/// Mutation: in `register_tiled_launch`, take the first candidate instead of `RegisterTiling::choose` when no pin is
/// set; the unpinned plan is no longer the cost model's choice.
#[test]
fn an_unpinned_plan_is_the_cost_models_choice_and_equals_the_plan_pinned_to_it() {
    let target = wgpu();
    let Target::Gpu { caps, .. } = &target else {
        unreachable!("a GPU target")
    };
    let chosen = RegisterTiling::choose(shape(), caps);
    let unpinned = compiled(&target, None).unwrap();
    assert_eq!(schedule_of(&unpinned), chosen.schedule());
    let pinned = compiled(&target, Some(chosen)).unwrap();
    assert_eq!(plan_text(&pinned), plan_text(&unpinned));
}

/// A pin the device cannot run is refused by name: a 256-lane tiling on a device that runs 64 invocations is
/// `PinnedTilingNotRunnable` naming that schedule, not the 64-lane tiling the cost model would take.
/// Mutation: in `register_tiled_launch`, drop the `runs_over` check; the pinned compile fails later, as
/// `KernelResources(WorkgroupDimension)`, and the assertion names it instead of `PinnedTilingNotRunnable`.
#[test]
fn a_pin_the_device_cannot_run_is_refused_by_name() {
    let wide = candidates(&wgpu())
        .into_iter()
        .find(|t| t.workgroup_lanes() == 256)
        .expect("the full device runs a 256-lane tiling");
    let target = narrow_workgroups();
    assert!(
        compiled(&target, None).is_ok(),
        "the narrow device plans the contraction unpinned"
    );
    match compiled(&target, Some(wide)) {
        Err(CompileError::Plan(error)) => match *error {
            PlanError::Refused(refusal) => assert_eq!(
                refusal.missing,
                Capability::PinnedTilingNotRunnable {
                    pinned: wide.schedule()
                }
            ),
            other => panic!("expected a typed refusal, got {other:?}"),
        },
        Err(other) => panic!("expected a typed refusal, got {other:?}"),
        Ok(program) => panic!("expected a refusal, planned {:?}", schedule_of(&program)),
    }
}
