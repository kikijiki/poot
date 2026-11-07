//! POOT-1505 (a1): a compile with register feedback asks the device's compiler whether a candidate tiling spills,
//! plans only tilings whose probe holds its registers, and names the compiler in the program; a compile without
//! feedback is the register model's plan, unchanged.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use poot_graph_ir::{Builder, Graph, MatmulPrecision, TensorType};
use poot_graph_plan::{
    Capability, CompileError, CompileLimits, CompileOptions, CompiledRegisters,
    FakeCompiledRegisters, FeedbackError, KernelChoice, PlanError, Program, RegisterFeedback,
    RegisterProbe, RegisterVerdict, Target, compile,
};
use poot_kernelgen::{ContractionSpec, KernelRequest, Schedule};
use poot_target::{
    AmdArch, Backend, DeviceCaps, KernelCompiler, MatrixUnits, Queried, SubgroupSupport,
};
use poot_tensor::DType;

const M: usize = 2048;
const K: usize = 1024;
const N: usize = 3072;

const CLANG: KernelCompiler = KernelCompiler::AmdClang {
    version: (23, 0, 0),
    commit: 0x4321_5c73_116c_407a,
    gfx: 0x1151,
};

fn rocm_wmma() -> Target {
    let mut caps = DeviceCaps::rocm_default();
    caps.matrix_units = Queried::Known(MatrixUnits::amd("gfx1151", 32));
    caps.register_file = AmdArch::gfx1151().register_file();
    caps.subgroup = Queried::Known(SubgroupSupport::Present {
        min_size: 32,
        max_size: 32,
    });
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps,
    }
}

/// `matmul(x, transpose(w))` over a bf16 `[N, K]` weight: one dense contraction.
fn dense_bf16() -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![M, K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![N, K], DType::BF16))
        .unwrap();
    let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    b.finish(out).unwrap()
}

/// A fake compiler: `spills(probe name)` decides, and the probes it was asked are counted.
fn compiler(
    spills: impl Fn(&str) -> bool + Send + Sync + 'static,
) -> (RegisterFeedback, Arc<AtomicUsize>) {
    #[derive(Debug)]
    struct Counted(FakeCompiledRegisters, Arc<AtomicUsize>);
    impl CompiledRegisters for Counted {
        fn compiler(&self) -> KernelCompiler {
            self.0.compiler()
        }
        fn compile(&self, probe: &RegisterProbe) -> Result<RegisterVerdict, FeedbackError> {
            self.1.fetch_add(1, Ordering::Relaxed);
            self.0.compile(probe)
        }
    }
    let asked = Arc::new(AtomicUsize::new(0));
    let fake = FakeCompiledRegisters {
        compiler: CLANG,
        verdict: Box::new(move |probe| {
            Ok(RegisterVerdict {
                registers: 128,
                local_bytes: if spills(&probe.body.name) { 64 } else { 0 },
            })
        }),
        compiles: Default::default(),
    };
    (
        RegisterFeedback::new(Counted(fake, Arc::clone(&asked))),
        asked,
    )
}

fn options(feedback: Option<RegisterFeedback>) -> CompileOptions {
    CompileOptions {
        register_feedback: feedback,
        ..CompileOptions::new(CompileLimits::STANDARD)
            .with_matmul_precision(MatmulPrecision::Narrow)
    }
}

/// The schedule of the program's dense contraction.
fn dense_schedule(program: &Program) -> Schedule {
    program
        .planned()
        .find_map(|(eqn, _)| match program.kernel_choice(eqn) {
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::DenseTiled(
                spec,
            )))
            | KernelChoice::Serialized {
                request: KernelRequest::Contraction(ContractionSpec::DenseTiled(spec)),
                ..
            } => Some(spec.schedule),
            _ => None,
        })
        .expect("a dense tiled contraction")
}

/// Without feedback the program is the register model's and names no compiler; with a compiler that holds every
/// probe the program names it and its tiling is one the compiler was asked about. Mutation: drop
/// `register_feedback` from `compile`'s `Program`; the feedback row reads `None`.
#[test]
fn a_program_planned_with_feedback_names_the_compiler() {
    let target = rocm_wmma();
    let model = compile(&dense_bf16(), &target, &options(None)).expect("the model plans it");
    assert_eq!(model.register_feedback(), None);
    let (feedback, asked) = compiler(|_| false);
    let program =
        compile(&dense_bf16(), &target, &options(Some(feedback))).expect("the compiler holds it");
    assert_eq!(program.register_feedback(), Some(CLANG));
    assert!(asked.load(Ordering::Relaxed) > 0, "the compiler was asked");
    assert!(matches!(
        dense_schedule(&program),
        Schedule::FragmentTiled { .. }
    ));
}

/// A compiler that spills every probe leaves the contraction no tiling: the plan refuses it by name rather than run a
/// tiling that spills. Mutation: fall back to `RegisterTiling::choose` in `register_tiled_launch` when no candidate
/// holds; the compile succeeds.
#[test]
fn a_contraction_no_tiling_of_which_holds_is_refused() {
    let (feedback, _) = compiler(|_| true);
    let error =
        compile(&dense_bf16(), &rocm_wmma(), &options(Some(feedback))).expect_err("nothing holds");
    let CompileError::Plan(plan) = error else {
        panic!("a plan refusal, got {error:?}");
    };
    let PlanError::Refused(refusal) = *plan else {
        panic!("a refusal, got {plan:?}");
    };
    assert!(
        matches!(refusal.missing, Capability::NoTilingHoldsItsRegisters),
        "{:?}",
        refusal.missing
    );
}

/// A compiler that spills the tiling it would otherwise plan makes the plan take the next one in cost order: the
/// tiling that runs is one the compiler holds. Mutation: in `RegisterTiling::cheapest`, return the cheapest candidate
/// whatever the compiler says; the program runs the spilling tiling.
#[test]
fn a_tiling_the_compiler_spills_gives_way_to_the_next() {
    let target = rocm_wmma();
    let (holds_all, _) = compiler(|_| false);
    let first =
        dense_schedule(&compile(&dense_bf16(), &target, &options(Some(holds_all))).unwrap());
    let refused = format!("{first:?}").replace(|c: char| !c.is_ascii_alphanumeric(), "_");
    let (feedback, _) = compiler(move |name| name.contains(&refused));
    let program =
        compile(&dense_bf16(), &target, &options(Some(feedback))).expect("another tiling holds");
    assert_ne!(dense_schedule(&program), first, "the first tiling spilled");
}
