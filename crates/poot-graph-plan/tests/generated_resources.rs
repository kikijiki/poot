//! Card 662: a planned kernel proves its launch resources and capability requirements against the
//! device's measured [`DeviceCaps`] before any loader sees it. Every row drives the production entry
//! (`compile`, which runs the one finalizer) with caps small enough that the kernel's needs and the
//! device's limits differ, then plays the program into a fake loader: a refused graph has no program, so
//! the loader's count stays zero. The kernel's resources are read here by walking the planned `Body`
//! independently of the validator, so a refusal's reported numbers are checked against a second reading.
//!
//! Generated kernels (a contraction, the tensor-core and cooperative-matrix fragment kernels) and an
//! imported one (the coalesced GEMV) take the same path and fail with the same typed refusal.

use poot_graph_ir::{Builder, TensorType};
use poot_graph_plan::device_validation::ResourceRefusal;
use poot_graph_plan::{
    Capability, CompileError, CompileLimits, CompileOptions, FusionPolicy, KernelChoice, Plan,
    PlanError, Program, Submission, Target, compile,
};
use poot_kernel_ir::{Body, Ty, WmmaDtype, WmmaShape};
use poot_kernelgen::{ContractionSpec, FragmentUse, KernelRequest};
use poot_target::{AmdArch, Backend, DeviceCaps, Queried, SubgroupSupport, TensorCoreSupport};
use poot_tensor::DType;
use poot_test_util::device_caps::default_caps_for;

fn options() -> CompileOptions {
    CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: CompileLimits::STANDARD,
    }
}

fn target(backend: Backend, caps: DeviceCaps) -> Target {
    Target { backend, caps }
}

/// A stand-in for the device's kernel loader: it records every body a program would hand to
/// `Device::load_kernel`, in the order the engine loads them.
#[derive(Default)]
struct FakeLoader {
    loaded: Vec<String>,
}

impl FakeLoader {
    fn load(&mut self, program: &Program) {
        for (_, plan) in program.planned() {
            match plan {
                Plan::Compute { key, .. } | Plan::ComputeMeta { key, .. } => {
                    self.loaded.push(key.clone());
                }
                Plan::ComputeChunks(chunks) => {
                    self.loaded
                        .extend(chunks.iter().map(|chunk| chunk.key.clone()));
                }
                Plan::Alias(_) | Plan::View { .. } | Plan::Collective { .. } => {}
            }
        }
    }
}

/// Compile `graph` and, when a program exists, load it. Returns the loader and the compile result.
fn compile_and_load(
    graph: &poot_graph_ir::Graph,
    target: &Target,
) -> (FakeLoader, Result<Program, CompileError>) {
    let result = compile(graph, target, &options());
    let mut loader = FakeLoader::default();
    if let Ok(program) = &result {
        loader.load(program);
    }
    (loader, result)
}

/// The typed resource refusal `result` carries, or a panic naming what it was instead.
fn resource_refusal(result: Result<Program, CompileError>, what: &str) -> ResourceRefusal {
    match result {
        Err(CompileError::Plan(error)) => match *error {
            PlanError::Refused(refusal) => match refusal.missing {
                Capability::KernelResources(resource) => resource,
                other => panic!("{what}: refused for {other:?}, not for kernel resources"),
            },
            other => panic!("{what}: planner error {other:?}"),
        },
        Err(other) => panic!("{what}: compile error {other:?}"),
        Ok(_) => panic!("{what}: the program compiled, so a loader would receive it"),
    }
}

/// Every dispatched body of `program`, with its launch grid.
fn dispatched(program: &Program) -> Vec<(&Body, [u32; 3])> {
    program
        .planned()
        .filter_map(|(_, plan)| match plan {
            Plan::Compute { body, grid, .. } | Plan::ComputeMeta { body, grid, .. } => {
                Some((body, *grid))
            }
            _ => None,
        })
        .collect()
}

/// The one dispatching plan of `program`.
fn the_plan(program: &Program) -> &Plan {
    let mut plans = program
        .planned()
        .map(|(_, plan)| plan)
        .filter(|plan| matches!(plan, Plan::Compute { .. } | Plan::ComputeMeta { .. }));
    let plan = plans.next().expect("a dispatching plan");
    assert!(
        plans.next().is_none(),
        "the fixture plans exactly one dispatch"
    );
    plan
}

/// The one dispatched body of `program`.
fn the_body(program: &Program) -> (&Body, [u32; 3]) {
    let mut bodies = dispatched(program);
    assert_eq!(bodies.len(), 1, "the fixture plans exactly one dispatch");
    bodies.remove(0)
}

/// Static workgroup memory read straight off the body's declarations, independent of the validator.
fn lds_bytes(body: &Body) -> u64 {
    body.workgroup_locals
        .iter()
        .map(|decl| {
            let width = match decl.elem_ty {
                Ty::F32 | Ty::I32 | Ty::U32 => 4,
                Ty::F16 | Ty::BF16 => 2,
                ref other => panic!("fixture LDS element {other:?} has no width here"),
            };
            width * u64::from(decl.len)
        })
        .sum()
}

fn invocations(body: &Body) -> u64 {
    body.workgroup_size.iter().map(|&e| u64::from(e)).product()
}

/// The generated tiled/serial contraction fixture: `[1, 8, 1024] @ [1024, 8]` f32 plans one generated
/// contraction with static workgroup memory on NVPTX.
fn generated_lds_graph() -> poot_graph_ir::Graph {
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![1, 8, 1024]));
    let w = b.constant("w", TensorType::f32(vec![1024, 8]));
    let out = b.matmul(a, w);
    b.finish(out)
}

/// The imported fixture: a decode GEMV (`M = 1`) on the wgpu target plans the shipped coalesced GEMV.
fn imported_gemv_graph() -> poot_graph_ir::Graph {
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![1, 1, 256]));
    let w = b.constant("w", TensorType::f32(vec![256, 512]));
    let out = b.matmul(a, w);
    b.finish(out)
}

fn choice_of(program: &Program) -> &KernelChoice {
    let (eqn, _) = program
        .planned()
        .find(|(_, plan)| matches!(plan, Plan::Compute { .. } | Plan::ComputeMeta { .. }))
        .expect("a dispatched equation");
    program.kernel_choice(eqn)
}

fn generated_contraction(program: &Program) -> &ContractionSpec {
    match choice_of(program) {
        KernelChoice::Generated(KernelRequest::Contraction(spec)) => spec,
        other => panic!("expected a generated contraction, got {other:?}"),
    }
}

/// SC-001, static LDS: the generated contraction's body declares `lds` bytes (read here independently).
/// A device offering exactly that admits it and loads it; one byte less refuses it with the exact figures
/// and the fake loader is never reached.
///
/// Mutation: in `validate_kernel_resources`, delete the `used.lds_bytes > caps.lds_bytes` check (or skip
/// `finalize`'s call to it). The one-byte-short compile then succeeds, `resource_refusal` panics with "the
/// program compiled, so a loader would receive it", and the loader holds the kernel.
#[test]
fn static_lds_over_the_budget_is_refused_before_any_load() {
    let backend = Backend::Nvptx;
    let roomy = target(backend, default_caps_for(backend));
    let (_, result) = compile_and_load(&generated_lds_graph(), &roomy);
    let program = result.expect("the roomy device plans it");
    generated_contraction(&program);
    let (body, _) = the_body(&program);
    let lds = lds_bytes(body);
    assert!(lds > 0, "the fixture must carry static workgroup memory");

    let mut exact = roomy;
    exact.caps.lds_bytes = u32::try_from(lds).unwrap();
    let (loader, result) = compile_and_load(&generated_lds_graph(), &exact);
    result.expect("a device offering exactly the declared bytes admits the kernel");
    assert_eq!(loader.loaded.len(), 1);

    let mut short = roomy;
    short.caps.lds_bytes = u32::try_from(lds).unwrap() - 1;
    let (loader, result) = compile_and_load(&generated_lds_graph(), &short);
    assert_eq!(
        resource_refusal(result, "one byte of LDS short"),
        ResourceRefusal::StaticLds {
            needed: lds,
            available: short.caps.lds_bytes,
        }
    );
    assert!(
        loader.loaded.is_empty(),
        "a refused plan reached the loader"
    );
}

/// SC-001 and SC-003, workgroup shape: an overlarge workgroup (total invocations, then one axis) is
/// refused for the generated contraction and for the imported GEMV alike, by the same finalizer, with the
/// body's own measured shape in the refusal.
///
/// Mutation: `finalize` stops calling `validate_planned_resources` (bypassing finalization). Every
/// compile below succeeds, `resource_refusal` panics, and the loader holds the kernels.
#[test]
fn an_overlarge_workgroup_is_refused_for_generated_and_imported_kernels_alike() {
    let fixtures = [
        ("generated", Backend::Nvptx, generated_lds_graph()),
        ("imported", Backend::SpirvVulkan, imported_gemv_graph()),
    ];
    for (what, backend, graph) in fixtures {
        let roomy = target(backend, default_caps_for(backend));
        let (_, result) = compile_and_load(&graph, &roomy);
        let program = result.unwrap_or_else(|error| panic!("{what}: {error}"));
        match (what, choice_of(&program)) {
            ("generated", KernelChoice::Generated(_))
            | ("imported", KernelChoice::Imported { .. }) => {}
            (_, other) => panic!("{what} fixture planned {other:?}"),
        }
        let (body, _) = the_body(&program);
        let total = invocations(body);
        let wg = body.workgroup_size;

        let mut too_many = roomy;
        too_many.caps.max_workgroup_invocations = u32::try_from(total).unwrap() - 1;
        let (loader, result) = compile_and_load(&graph, &too_many);
        assert_eq!(
            resource_refusal(result, what),
            ResourceRefusal::WorkgroupInvocations {
                invocations: total,
                limit: too_many.caps.max_workgroup_invocations,
            },
            "{what}"
        );
        assert!(
            loader.loaded.is_empty(),
            "{what}: a refused plan reached the loader"
        );

        let mut too_wide = roomy;
        too_wide.caps.max_workgroup_size[0] = wg[0] - 1;
        let (loader, result) = compile_and_load(&graph, &too_wide);
        assert_eq!(
            resource_refusal(result, what),
            ResourceRefusal::WorkgroupDimension {
                axis: 0,
                size: wg[0],
                limit: wg[0] - 1,
            },
            "{what}"
        );
        assert!(
            loader.loaded.is_empty(),
            "{what}: a refused plan reached the loader"
        );

        let mut exact = roomy;
        exact.caps.max_workgroup_invocations = u32::try_from(total).unwrap();
        exact.caps.max_workgroup_size[0] = wg[0];
        let (loader, result) = compile_and_load(&graph, &exact);
        result.unwrap_or_else(|error| panic!("{what}: exact limits admit the kernel: {error}"));
        assert_eq!(loader.loaded.len(), 1, "{what}");
    }
}

/// SC-003, imported static LDS: the shipped GEMV's workgroup memory is held to the device budget by the
/// same finalizer as a generated body.
///
/// Mutation: `finalize` skips imported choices (return early for `KernelChoice::Imported`) and the
/// short-by-one compile succeeds.
#[test]
fn an_imported_kernel_over_the_lds_budget_is_refused_before_any_load() {
    let backend = Backend::SpirvVulkan;
    let roomy = target(backend, default_caps_for(backend));
    let (_, result) = compile_and_load(&imported_gemv_graph(), &roomy);
    let program = result.expect("the roomy device plans it");
    assert!(matches!(choice_of(&program), KernelChoice::Imported { .. }));
    let lds = lds_bytes(the_body(&program).0);
    assert!(lds > 0);

    let mut short = roomy;
    short.caps.lds_bytes = u32::try_from(lds).unwrap() - 1;
    let (loader, result) = compile_and_load(&imported_gemv_graph(), &short);
    assert_eq!(
        resource_refusal(result, "imported, one byte of LDS short"),
        ResourceRefusal::StaticLds {
            needed: lds,
            available: short.caps.lds_bytes,
        }
    );
    assert!(loader.loaded.is_empty());
}

/// A matmul whose operands are 16-aligned bf16 and whose output is f32 plans the matrix-fragment kernel
/// on a WMMA-capable AMD arch.
fn bf16_fragment_graph() -> poot_graph_ir::Graph {
    let b = Builder::new();
    let a = b.constant("a", TensorType::new(vec![1, 16, 32], DType::BF16));
    let w = b.constant("w", TensorType::new(vec![32, 16], DType::BF16));
    let out = b.matmul(a, w);
    let mut graph = b.finish(out);
    let out = graph.output;
    graph.values[out].aval.dtype = DType::F32;
    graph
}

/// One degraded-device row: what changed, how, and the refusal it must produce.
type DegradedRow = (&'static str, Box<dyn Fn(&mut DeviceCaps)>, ResourceRefusal);

/// SC-001 and SC-003, fragment and subgroup features: the tensor-core kernel declares a 32-lane subgroup
/// and uses a bf16 16x16x16 fragment. A device that reports no subgroups, subgroups of another size, no
/// matrix hardware, or matrix hardware its API does not expose, refuses it with the matching typed
/// reason; `Unknown` is never read as "present". The device that reports 32-lane subgroups and the WMMA
/// family admits it.
///
/// Mutation: in `validate_kernel_resources`, replace the `match caps.subgroup` with `Ok` for `Unknown`
/// (treat an unreported capability as present) and the `Unknown` row fails with "the program compiled".
#[test]
fn fragment_kernels_need_the_subgroup_and_matrix_hardware_the_device_reports() {
    let arch = AmdArch::gfx1151();
    let backend = Backend::AmdGcn(arch);
    let roomy = target(backend, default_caps_for(backend));
    let graph = bf16_fragment_graph();
    let (loader, result) = compile_and_load(&graph, &roomy);
    let program = result.expect("gfx1151 reports wave32 and WMMA");
    assert!(matches!(
        generated_contraction(&program),
        ContractionSpec::TensorCore { .. }
    ));
    let (body, _) = the_body(&program);
    assert!(
        invocations(body).is_multiple_of(32),
        "a fragment kernel's workgroup is whole subgroups"
    );
    assert_eq!(loader.loaded.len(), 1);

    let rows: [DegradedRow; 5] = [
        (
            "no subgroup support",
            Box::new(|caps| caps.subgroup = Queried::Known(SubgroupSupport::Absent)),
            ResourceRefusal::SubgroupAbsent { lanes: 32 },
        ),
        (
            "subgroup support not exposed",
            Box::new(|caps| caps.subgroup = Queried::Unknown),
            ResourceRefusal::SubgroupUnknown { lanes: 32 },
        ),
        (
            "64-lane subgroups only",
            Box::new(|caps| {
                caps.subgroup = Queried::Known(SubgroupSupport::Present {
                    min_size: 64,
                    max_size: 64,
                });
            }),
            ResourceRefusal::SubgroupSize {
                lanes: 32,
                min_size: 64,
                max_size: 64,
            },
        ),
        (
            "no matrix hardware",
            Box::new(|caps| caps.tensor_core = TensorCoreSupport::None),
            ResourceRefusal::FragmentUnsupported {
                fragment: FragmentUse {
                    dtype: WmmaDtype::Bf16,
                    shape: WmmaShape::M16N16K16,
                },
                tensor_core: TensorCoreSupport::None,
            },
        ),
        (
            "matrix hardware not exposed",
            Box::new(|caps| caps.tensor_core = TensorCoreSupport::UnknownNotExposedByApi),
            ResourceRefusal::FragmentCapabilityUnknown {
                fragment: FragmentUse {
                    dtype: WmmaDtype::Bf16,
                    shape: WmmaShape::M16N16K16,
                },
            },
        ),
    ];
    for (what, degrade, expected) in rows {
        let mut degraded = roomy;
        degrade(&mut degraded.caps);
        let (loader, result) = compile_and_load(&graph, &degraded);
        assert_eq!(resource_refusal(result, what), expected, "{what}");
        assert!(
            loader.loaded.is_empty(),
            "{what}: a refused plan reached the loader"
        );
    }
}

/// The wgpu cooperative-matrix kernel: one 32-lane subgroup per 16x16 tile, f16 operands.
fn f16_coopmat_graph() -> poot_graph_ir::Graph {
    let b = Builder::new();
    let a = b.constant("a", TensorType::new(vec![1, 16, 16], DType::F16));
    let w = b.constant("w", TensorType::new(vec![16, 16], DType::F16));
    let out = b.matmul(a, w);
    let mut graph = b.finish(out);
    let out = graph.output;
    graph.values[out].aval.dtype = DType::F32;
    graph
}

/// SC-001, the same rule for the SPIR-V cooperative-matrix kernel: a Vulkan device with no subgroup
/// support refuses it before load. Mutation: as above (skip the subgroup check).
#[test]
fn the_cooperative_matrix_kernel_is_refused_without_subgroups() {
    let backend = Backend::SpirvVulkan;
    let roomy = target(backend, default_caps_for(backend));
    let graph = f16_coopmat_graph();
    let (loader, result) = compile_and_load(&graph, &roomy);
    let program = result.expect("the RADV fixture reports subgroups and WMMA");
    assert!(matches!(
        generated_contraction(&program),
        ContractionSpec::Coopmat { .. }
    ));
    assert_eq!(loader.loaded.len(), 1);

    let mut no_subgroups = roomy;
    no_subgroups.caps.subgroup = Queried::Known(SubgroupSupport::Absent);
    let (loader, result) = compile_and_load(&graph, &no_subgroups);
    assert_eq!(
        resource_refusal(result, "no subgroups"),
        ResourceRefusal::SubgroupAbsent { lanes: 32 }
    );
    assert!(loader.loaded.is_empty());
}

/// `[1, h, m, k] @ [1, h, k, n]` f32: two batched operands (no shared weight), which no tiled or GEMV
/// kernel takes, so the planner picks the one-thread-per-output serial kernel.
fn serial_contraction_graph(h: usize, m: usize, k: usize, n: usize) -> poot_graph_ir::Graph {
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![1, h, m, k]));
    let w = b.constant("w", TensorType::f32(vec![1, h, k, n]));
    let out = b.matmul(a, w);
    b.finish(out)
}

/// SC-002: a long serial contraction under a deliberately small per-dispatch budget. `threads * K`
/// serial steps is the dispatch's conservative work; a device bounding a dispatch at exactly that admits
/// it, one step less refuses it before any loader (the executor cannot batch around a single dispatch
/// that is itself too long), and the same budget admits the shorter contraction the budget was not
/// meant to stop.
///
/// Mutation: delete the `caps.max_dispatch_work` check in `validate_kernel_resources`; the over-budget
/// compile succeeds and `resource_refusal` panics with "the program compiled".
#[test]
fn a_long_serial_contraction_over_the_dispatch_budget_is_refused_before_dispatch() {
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let roomy = target(backend, default_caps_for(backend));
    let (h, m, k, n) = (2, 3, 4096, 5);
    let graph = serial_contraction_graph(h, m, k, n);
    let (_, result) = compile_and_load(&graph, &roomy);
    let program = result.expect("the roomy device plans it");
    assert!(
        matches!(
            generated_contraction(&program),
            ContractionSpec::Serial { .. }
        ),
        "the fixture must take the serial kernel"
    );
    let work = (h * m * n * k) as u64;

    let mut exact = roomy;
    exact.caps.max_dispatch_work = Queried::Known(work);
    let (loader, result) = compile_and_load(&graph, &exact);
    result.expect("a budget equal to the dispatch's work admits it");
    assert_eq!(loader.loaded.len(), 1);

    let mut small = roomy;
    small.caps.max_dispatch_work = Queried::Known(work - 1);
    let (loader, result) = compile_and_load(&graph, &small);
    assert_eq!(
        resource_refusal(result, "one step over the budget"),
        ResourceRefusal::DispatchWork {
            work,
            budget: work - 1,
        }
    );
    assert!(loader.loaded.is_empty());

    let short = serial_contraction_graph(h, m, k / 64, n);
    let (loader, result) = compile_and_load(&short, &small);
    result.expect("the same budget admits the short contraction");
    assert_eq!(loader.loaded.len(), 1);
}

/// `a + b` over `n` f32 elements: one thread per element, so a device whose grid cap is below
/// `n / 64` workgroups gets its workgroup widened to 256 by `finalize`.
fn add_graph(n: usize) -> poot_graph_ir::Graph {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![n]));
    let y = b.constant("y", TensorType::f32(vec![n]));
    let out = b.binary(poot_graph_ir::BinOp::Add, x, y);
    b.finish(out)
}

/// The finalizer's workgroup bump changes the kernel a request builds, so the plan key must change with
/// it: two devices whose grid caps differ plan different bodies for the same request and must never share
/// a key (a kernel cache keyed by it would serve one device the other's body).
///
/// Mutation: delete the `key.push_str(..)` in `shape_launch`. The widened plan then keeps the unwidened
/// plan's key, and the key assertion fails with equal keys beside different workgroup shapes.
#[test]
fn a_finalizer_workgroup_bump_is_reflected_in_the_plan_key() {
    let backend = Backend::SpirvVulkan;
    let n = 1024 * 64 + 1;
    let plain = target(backend, default_caps_for(backend));
    let mut tight = plain;
    tight.caps.max_grid = [1024; 3];

    let plain_program = compile(&add_graph(n), &plain, &options()).unwrap();
    let tight_program = compile(&add_graph(n), &tight, &options()).unwrap();
    let key_and_wg = |program: &Program| match the_plan(program) {
        Plan::Compute { body, key, .. } => (key.clone(), body.workgroup_size),
        other => panic!("expected Plan::Compute, got {other:?}"),
    };
    let (plain_key, plain_wg) = key_and_wg(&plain_program);
    let (tight_key, tight_wg) = key_and_wg(&tight_program);
    assert_eq!(
        plain_wg[0], 64,
        "the roomy device keeps the stock workgroup"
    );
    assert_eq!(
        tight_wg[0], 256,
        "the tight device widens it to fit the grid cap"
    );
    assert_ne!(
        plain_key, tight_key,
        "two different bodies ({plain_wg:?} and {tight_wg:?}) share the plan key {plain_key}"
    );
}

/// `a [1, m, k] f32 @ w [n, k] bf16 ^T`: folded by `compile` into the dense BF16 contraction, whose
/// `M > 1` arm plans the imported one-thread-per-output body.
fn dense_bf16_graph(m: usize, k: usize, n: usize) -> poot_graph_ir::Graph {
    let b = Builder::new();
    let a = b.constant("a", TensorType::new(vec![1, m, k], DType::F32));
    let w = b.constant("w", TensorType::new(vec![n, k], DType::BF16));
    let out = b.matmul(a, b.transpose(w, vec![1, 0]));
    b.finish(out)
}

fn imported_kernel(program: &Program) -> poot_graph_plan::ImportedKernel {
    match choice_of(program) {
        KernelChoice::Imported { kernel, .. } => *kernel,
        other => panic!("expected an imported kernel, got {other:?}"),
    }
}

/// Imported contractions loop over K per thread, so they declare serial work and a device's
/// `max_dispatch_work` bounds them like a generated serial contraction: `output elements * K` steps. A
/// budget equal to the work admits the imported kernel; one step less either refuses the plan (the
/// coalesced GEMV, no alternative) or sends the equation down its primitive lowering, which no longer
/// names the imported kernel (the dense BF16 contraction, whose fold `compile` only keeps while the folded
/// plan is legal).
///
/// Mutation: in `ImportedKernel::needs`, move the contraction family to `ImportedNeeds::NONE`; the
/// one-step-short compile still plans the imported kernel and the row fails ("the program compiled" or
/// "still plans the imported kernel").
#[test]
fn imported_contractions_are_bounded_by_the_dispatch_work_budget() {
    let gemv = (
        Backend::SpirvVulkan,
        imported_gemv_graph(),
        512u64 * 256,
        true,
    );
    let dense = (
        Backend::Nvptx,
        dense_bf16_graph(2, 64, 32),
        2 * 32 * 64,
        false,
    );
    for (what, (backend, graph, work, refuses)) in
        [("coalesced gemv", gemv), ("dense bf16 M>1", dense)]
    {
        let roomy = target(backend, default_caps_for(backend));
        let (_, result) = compile_and_load(&graph, &roomy);
        let program = result.unwrap_or_else(|error| panic!("{what}: {error}"));
        let imported = imported_kernel(&program);

        let mut exact = roomy;
        exact.caps.max_dispatch_work = Queried::Known(work);
        let (loader, result) = compile_and_load(&graph, &exact);
        result.unwrap_or_else(|error| {
            panic!("{what}: a budget equal to the work admits it: {error}")
        });
        assert_eq!(loader.loaded.len(), 1, "{what}");

        let mut short = roomy;
        short.caps.max_dispatch_work = Queried::Known(work - 1);
        let (loader, result) = compile_and_load(&graph, &short);
        if refuses {
            assert_eq!(
                resource_refusal(result, what),
                ResourceRefusal::DispatchWork {
                    work,
                    budget: work - 1,
                },
                "{what}"
            );
            assert!(loader.loaded.is_empty(), "{what}");
        } else {
            let program = result.unwrap_or_else(|error| panic!("{what}: {error}"));
            let still = program.planned().any(|(eqn, _)| {
                matches!(program.kernel_choice(eqn),
                    KernelChoice::Imported { kernel, .. } if *kernel == imported)
            });
            assert!(
                !still,
                "{what}: one step over the budget still plans the imported kernel"
            );
        }
    }
}
