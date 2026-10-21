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
use poot_graph_plan::test_support::{gpu_caps, gpu_caps_mut};
use poot_graph_plan::{
    Capability, CompileError, CompileLimits, CompileOptions, KernelChoice, Plan, PlanError,
    Program, Target, compile,
};
use poot_kernel_ir::{Body, Ty, WmmaDtype, WmmaShape};
use poot_kernelgen::{ContractionSpec, FragmentUse, KernelRequest};
use poot_target::{AmdArch, Backend, DeviceCaps, Queried, SubgroupSupport, TensorCoreSupport};
use poot_tensor::DType;
use poot_test_util::device_caps::default_caps_for;

fn options() -> CompileOptions {
    CompileOptions::new(CompileLimits::STANDARD)
}

fn target(backend: Backend, caps: DeviceCaps) -> Target {
    Target::Gpu { backend, caps }
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
            self.loaded.extend(
                plan.dispatches()
                    .iter()
                    .map(|dispatch| dispatch.key().to_string()),
            );
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
        .filter_map(|(_, plan)| plan.single())
        .map(|dispatch| (dispatch.body(), dispatch.launch().grid))
        .collect()
}

/// The one dispatching plan of `program`.
fn the_plan(program: &Program) -> &Plan {
    let mut plans = program
        .planned()
        .map(|(_, plan)| plan)
        .filter(|plan| plan.single().is_some());
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
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![1, 8, 1024])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![1024, 8])).unwrap();
    let out = b.matmul(a, w).unwrap();
    b.finish(out).unwrap()
}

/// The imported fixture: a decode GEMV (`M = 1`) on the wgpu target plans the shipped coalesced GEMV.
fn imported_gemv_graph() -> poot_graph_ir::Graph {
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![1, 1, 256])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![256, 512])).unwrap();
    let out = b.matmul(a, w).unwrap();
    b.finish(out).unwrap()
}

fn choice_of(program: &Program) -> &KernelChoice {
    let (eqn, _) = program
        .planned()
        .find(|(_, plan)| plan.single().is_some())
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
    gpu_caps_mut(&mut exact).lds_bytes = u32::try_from(lds).unwrap();
    let (loader, result) = compile_and_load(&generated_lds_graph(), &exact);
    result.expect("a device offering exactly the declared bytes admits the kernel");
    assert_eq!(loader.loaded.len(), 1);

    let mut short = roomy;
    gpu_caps_mut(&mut short).lds_bytes = u32::try_from(lds).unwrap() - 1;
    let (loader, result) = compile_and_load(&generated_lds_graph(), &short);
    assert_eq!(
        resource_refusal(result, "one byte of LDS short"),
        ResourceRefusal::StaticLds {
            needed: lds,
            available: gpu_caps(&short).lds_bytes,
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
        gpu_caps_mut(&mut too_many).max_workgroup_invocations = u32::try_from(total).unwrap() - 1;
        let (loader, result) = compile_and_load(&graph, &too_many);
        assert_eq!(
            resource_refusal(result, what),
            ResourceRefusal::WorkgroupInvocations {
                invocations: total,
                limit: gpu_caps(&too_many).max_workgroup_invocations,
            },
            "{what}"
        );
        assert!(
            loader.loaded.is_empty(),
            "{what}: a refused plan reached the loader"
        );

        let mut too_wide = roomy;
        gpu_caps_mut(&mut too_wide).max_workgroup_size[0] = wg[0] - 1;
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
        gpu_caps_mut(&mut exact).max_workgroup_invocations = u32::try_from(total).unwrap();
        gpu_caps_mut(&mut exact).max_workgroup_size[0] = wg[0];
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
    gpu_caps_mut(&mut short).lds_bytes = u32::try_from(lds).unwrap() - 1;
    let (loader, result) = compile_and_load(&imported_gemv_graph(), &short);
    assert_eq!(
        resource_refusal(result, "imported, one byte of LDS short"),
        ResourceRefusal::StaticLds {
            needed: lds,
            available: gpu_caps(&short).lds_bytes,
        }
    );
    assert!(loader.loaded.is_empty());
}

/// A matmul whose operands are 16-aligned bf16 and whose output is f32 plans the matrix-fragment kernel
/// on a WMMA-capable AMD arch.
fn bf16_fragment_graph() -> poot_graph_ir::Graph {
    let b = Builder::for_tests();
    let a = b
        .constant("a", TensorType::new(vec![1, 16, 32], DType::BF16))
        .unwrap();
    let w = b
        .constant("w", TensorType::new(vec![32, 16], DType::BF16))
        .unwrap();
    let out = b.matmul_f32_accumulate(a, w).unwrap();
    b.finish(out).unwrap()
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
        degrade(gpu_caps_mut(&mut degraded));
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
    let b = Builder::for_tests();
    let a = b
        .constant("a", TensorType::new(vec![1, 16, 16], DType::F16))
        .unwrap();
    let w = b
        .constant("w", TensorType::new(vec![16, 16], DType::F16))
        .unwrap();
    let out = b.matmul_f32_accumulate(a, w).unwrap();
    b.finish(out).unwrap()
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
    gpu_caps_mut(&mut no_subgroups).subgroup = Queried::Known(SubgroupSupport::Absent);
    let (loader, result) = compile_and_load(&graph, &no_subgroups);
    assert_eq!(
        resource_refusal(result, "no subgroups"),
        ResourceRefusal::SubgroupAbsent { lanes: 32 }
    );
    assert!(loader.loaded.is_empty());
}

/// `a + b` over `n` f32 elements: one thread per element, so a device whose grid cap is below
/// `n / 64` workgroups gets its workgroup widened to 256 by `finalize`.
fn add_graph(n: usize) -> poot_graph_ir::Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![n])).unwrap();
    let y = b.constant("y", TensorType::f32(vec![n])).unwrap();
    let out = b.binary(poot_graph_ir::BinOp::Add, x, y).unwrap();
    b.finish(out).unwrap()
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
    gpu_caps_mut(&mut tight).max_grid = [1024; 3];

    let plain_program = compile(&add_graph(n), &plain, &options()).unwrap();
    let tight_program = compile(&add_graph(n), &tight, &options()).unwrap();
    let key_and_wg = |program: &Program| match the_plan(program).single() {
        Some(dispatch) if dispatch.launch().meta.is_empty() => {
            (dispatch.key().to_string(), dispatch.body().workgroup_size)
        }
        _ => panic!("expected one plain dispatch, got {:?}", the_plan(program)),
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
