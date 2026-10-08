//! The recurrence scan composite (POOT-634) through compile, for both families: a generated schedule whose
//! body does not grow with the sequence, a launch count bounded by `max_dispatch_work`, distinct buffers for
//! `y` and `s_out`, and `s_out` published as state. No device.

use poot_executor_parity::dense::step;
use poot_graph_ir::{Graph, OpKind, ValidationOutputs};
use poot_graph_plan::device_validation::ResourceRefusal;
use poot_graph_plan::{
    Capability, CompileError, CompileLimits, CompileOptions, KernelChoice, Plan, PlanError,
    Program, Submission, Target, compile,
};
use poot_kernelgen::{KernelRequest, ScanKernel, ScanSpec};
use poot_models::model::{LogitRows, Phase};
use poot_models::registry::Registry;
use poot_target::{AmdArch, Backend, DeviceCaps, Queried};
use poot_test_util::scan::ScanFixture;

fn targets() -> [(&'static str, Target); 3] {
    [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ]
    .map(|backend| {
        (
            match backend {
                Backend::SpirvVulkan => "spirv",
                Backend::Nvptx => "nvptx",
                _ => "amdgcn",
            },
            Target {
                backend,
                caps: poot_test_util::device_caps::default_caps_for(backend),
            },
        )
    })
}

fn compiled(graph: &Graph, target: &Target) -> Result<Program, CompileError> {
    compile(
        graph,
        target,
        &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
    )
}

fn families(len: usize) -> [ScanFixture; 2] {
    [
        ScanFixture::delta_rule(2, len, 8, 8),
        ScanFixture::ssd(2, len, 8, 4, 8),
    ]
}

/// The generated scan request, and the scan equation's instruction counts per dispatch.
fn scan_plan(program: &Program) -> (ScanSpec, Vec<usize>) {
    let (eqn, plan) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::Scan(_)))
        .expect("the compiled program keeps the scan");
    let KernelChoice::Generated(KernelRequest::Scan(spec)) = program.kernel_choice(eqn) else {
        panic!(
            "the scan is not generated: {:?}",
            program.kernel_choice(eqn)
        );
    };
    let Plan::Dispatches(dispatches) = plan else {
        panic!("the scan launches: {plan:?}");
    };
    let sizes = dispatches
        .iter()
        .map(|d| {
            d.body
                .blocks
                .iter()
                .map(|block| block.statements.len() + 1)
                .sum()
        })
        .collect();
    (spec.clone(), sizes)
}

/// The traced graph holds one scan equation and the generated body is the same size for every sequence
/// length: the sequence loop is a runtime loop, not one body copy per chunk.
#[test]
fn graph_and_body_do_not_grow_with_the_chunk_count() {
    let lens = [8, 64, 4096];
    for (family, make) in [
        (
            "delta_rule",
            (|len| families(len)[0]) as fn(usize) -> ScanFixture,
        ),
        ("ssd", |len| families(len)[1]),
    ] {
        let equations: Vec<usize> = lens.map(|len| make(len).graph().graph.eqns.len()).into();
        assert!(
            equations.windows(2).all(|w| w[0] == w[1]),
            "{family}: the traced graph grows with the sequence: equations at L={lens:?} are {equations:?}"
        );
        for (name, target) in targets() {
            let sizes: Vec<_> = lens
                .map(|len| {
                    let program = compiled(&make(len).graph().graph, &target)
                        .unwrap_or_else(|e| panic!("{family} {name} L={len}: {e:?}"));
                    scan_plan(&program).1
                })
                .into();
            eprintln!("{family} {name}: generated scan body instructions at L={lens:?}: {sizes:?}");
            assert!(
                sizes.windows(2).all(|w| w[0] == w[1]),
                "{family} {name}: the generated body grows with the sequence: instructions at L={lens:?} are {sizes:?}"
            );
        }
    }
}

/// The bodies of the scan equation's dispatches, in launch order.
fn scan_bodies(program: &Program) -> Vec<poot_kernel_ir::Body> {
    let (_, plan) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::Scan(_)))
        .expect("the compiled program keeps the scan");
    let Plan::Dispatches(dispatches) = plan else {
        panic!("the scan launches: {plan:?}");
    };
    dispatches.iter().map(|d| d.body.clone()).collect()
}

/// `y` and `s_out` are distinct values with distinct buffers, both distinct from the packed result they
/// project, and `s_out` is the committed state.
#[test]
fn results_have_distinct_buffers_and_s_out_is_committed() {
    for fixture in families(6) {
        for (name, target) in targets() {
            let program = compiled(&fixture.graph().graph, &target).unwrap();
            let graph = program.graph();
            let packed = graph
                .eqns
                .iter()
                .find(|e| matches!(e.op, OpKind::Scan(_)))
                .unwrap()
                .out;
            let y = graph.output;
            let (s_in, s_out) = graph.state[0];
            let plan = program.buffer_plan();
            let slots = [packed, y, s_out].map(|v| plan.slot(v));
            let label = format!("{:?} {name}", fixture.family);
            assert!(
                slots.iter().all(Option::is_some),
                "{label}: every result is arena-backed: {slots:?}"
            );
            assert_ne!(
                slots[0], slots[1],
                "{label}: y does not alias the packed result"
            );
            assert_ne!(
                slots[0], slots[2],
                "{label}: s_out does not alias the packed result"
            );
            assert_ne!(
                slots[1], slots[2],
                "{label}: y and s_out are distinct buffers"
            );
            assert!(
                program
                    .state_commit()
                    .copies
                    .iter()
                    .any(|c| c.state_output == s_out && c.state_input == s_in),
                "{label}: s_out is committed over s_in: {:?}",
                program.state_commit()
            );
        }
    }
}

fn refusal(error: CompileError) -> Capability {
    let CompileError::Plan(error) = error else {
        panic!("expected a plan refusal, got {error:?}");
    };
    let PlanError::Refused(refusal) = *error else {
        panic!("expected a typed refusal, got {error:?}");
    };
    refusal.missing
}

/// A state column that cannot fit workgroup memory is refused at plan time naming the capability.
#[test]
fn a_state_column_that_cannot_fit_is_refused_before_dispatch() {
    let (_, mut target) = targets().into_iter().next().unwrap();
    target.caps.lds_bytes = 64; // 16 floats: a 32-row column does not fit
    for fixture in [
        ScanFixture::delta_rule(1, 4, 32, 4),
        ScanFixture::ssd(1, 4, 4, 32, 4),
    ] {
        let missing = refusal(compiled(&fixture.graph().graph, &target).unwrap_err());
        assert_eq!(
            missing,
            Capability::ScanLds {
                rows: 32,
                lds_bytes: 64
            },
            "{:?}",
            fixture.family
        );
    }
}

fn budgeted(target: &Target, budget: u64) -> Target {
    Target {
        backend: target.backend,
        caps: DeviceCaps {
            max_dispatch_work: Queried::Known(budget),
            ..target.caps
        },
    }
}

/// A device without a calibrated bound runs the scan as one launch; a known `max_dispatch_work` splits it
/// into launches that each stay within it and together cover every token; a bound below one token refuses
/// naming the work, before any dispatch.
#[test]
fn the_work_bound_splits_the_sequence_or_refuses_before_dispatch() {
    for fixture in families(48) {
        for (name, target) in targets() {
            let label = format!("{:?} {name}", fixture.family);
            let (whole, launches) = scan_plan(&compiled(&fixture.graph().graph, &target).unwrap());
            assert_eq!(
                launches.len(),
                1,
                "{label}: an unbounded device launches once"
            );
            assert_eq!((whole.phase.start, whole.phase.count), (0, 48), "{label}");
            assert_eq!(
                whole.kernel,
                match fixture.family {
                    poot_graph_ir::ScanFamily::DeltaRule => ScanKernel::DeltaRule,
                    poot_graph_ir::ScanFamily::Ssd => ScanKernel::Ssd,
                }
            );

            let token_work = {
                let one = ScanSpec {
                    phase: poot_kernelgen::ScanPhase { start: 0, count: 1 },
                    ..whole.clone()
                };
                poot_kernelgen::KernelRequest::Scan(one)
                    .requirements()
                    .serial_work
                    .unwrap()
                    .total()
            };
            let budget = token_work * 10; // ten tokens per launch
            let (first, launches) =
                scan_plan(&compiled(&fixture.graph().graph, &budgeted(&target, budget)).unwrap());
            assert_eq!(
                first.phase.count, 10,
                "{label}: tokens per launch under budget {budget}"
            );
            assert_eq!(
                launches.len(),
                48usize.div_ceil(10),
                "{label}: launches cover the sequence"
            );
            // Launch `k` covers tokens `10k..` : its body is the body generated for exactly that phase.
            let bodies =
                scan_bodies(&compiled(&fixture.graph().graph, &budgeted(&target, budget)).unwrap());
            for (k, body) in bodies.iter().enumerate() {
                let start = k * first.phase.count;
                let expected = ScanSpec {
                    phase: poot_kernelgen::ScanPhase {
                        start,
                        count: first.phase.count.min(48 - start),
                    },
                    ..first.clone()
                };
                let generated = poot_kernelgen::generate(
                    &KernelRequest::Scan(expected),
                    &target.caps,
                    &poot_kernelgen::BodyLimits {
                        max_instructions: std::num::NonZeroUsize::new(1 << 16).unwrap(),
                        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
                    },
                )
                .unwrap();
                assert_eq!(
                    body, &generated.body,
                    "{label}: launch {k} covers tokens {start}.."
                );
            }
            let work = poot_kernelgen::KernelRequest::Scan(first)
                .requirements()
                .serial_work
                .unwrap()
                .total();
            assert!(
                work <= budget,
                "{label}: a launch carries {work} > {budget}"
            );

            let missing = refusal(
                compiled(&fixture.graph().graph, &budgeted(&target, token_work - 1)).unwrap_err(),
            );
            assert!(
                matches!(missing, Capability::KernelResources(ResourceRefusal::DispatchWork { work, .. }) if work == token_work),
                "{label}: a bound below one token refuses naming {token_work}: {missing:?}"
            );
        }
    }
}

/// Small and large per-head states: the planner picks the widest workgroup whose state columns fit the device's
/// workgroup memory, and the body it plans uses exactly the workgroup memory the request declares (measured
/// from the body, so a declaration cannot understate it) and never more than the device offers.
#[test]
fn per_head_states_stay_within_the_declared_resources() {
    for (name, target) in targets() {
        for rows in [4usize, 32, 128, 256] {
            for fixture in [
                ScanFixture::delta_rule(2, 16, rows, 8),
                ScanFixture::ssd(2, 16, 8, rows, 8),
            ] {
                let label = format!("{:?} rows={rows} on {name}", fixture.family);
                let program = compiled(&fixture.graph().graph, &target)
                    .unwrap_or_else(|e| panic!("{label}: {e:?}"));
                let (spec, _) = scan_plan(&program);
                let (_, plan) = program
                    .planned()
                    .find(|(eqn, _)| matches!(eqn.op, OpKind::Scan(_)))
                    .unwrap();
                let Plan::Dispatches(dispatches) = plan else {
                    unreachable!("asserted above")
                };
                let used = poot_kernelgen::BodyUse::measure(&dispatches[0].body)
                    .expect("measurable workgroup memory");
                assert_eq!(
                    used.lds_bytes,
                    spec.lds_bytes(),
                    "{label}: declared against measured LDS"
                );
                assert!(
                    used.lds_bytes <= u64::from(target.caps.lds_bytes),
                    "{label}: LDS {} over the device",
                    used.lds_bytes
                );
                assert_eq!(
                    used.workgroup,
                    [spec.cols as u32, 1, 1],
                    "{label}: workgroup"
                );
                assert!(
                    4 * rows * (spec.cols * 2) > target.caps.lds_bytes as usize
                        || spec.cols == spec.width.min(64),
                    "{label}: {} columns is not the widest that fits",
                    spec.cols
                );
            }
        }
    }
}

// ---- the model prefills the scan composite is emitted into ------------------------------------------------

const PROMPT: usize = 4;

fn registered(
    family: &str,
    phase: Phase,
    rows: usize,
    tokens: usize,
    cap: usize,
) -> Graph<ValidationOutputs> {
    let registry = Registry::builtin().expect("the builtin registry");
    // `family/variant` names one of the family's option fixtures (llama's Mixtral, granite's granitemoe).
    let (name, variant) = family.split_once('/').unwrap_or((family, ""));
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == name)
        .unwrap_or_else(|| panic!("no registered family {name}"));
    let fixture = match variant {
        "" => (entry.fixture)(),
        variant => {
            let (_, make) = entry
                .variants
                .iter()
                .find(|(n, _)| *n == variant)
                .unwrap_or_else(|| panic!("{name} has no fixture variant {variant}"));
            make()
        }
    };
    let model =
        (entry.build)(&fixture.raw(), &fixture.store).unwrap_or_else(|e| panic!("{family}: {e}"));
    model
        .trace(phase, step(rows, tokens, cap, LogitRows::Last))
        .unwrap_or_else(|e| panic!("{family}: {e}"))
}

/// The scans of `program` and, for each, the generated request and the dispatches it launches.
fn scans(program: &Program<ValidationOutputs>) -> Vec<(ScanSpec, usize)> {
    program
        .planned()
        .filter(|(eqn, _)| matches!(eqn.op, OpKind::Scan(_)))
        .map(|(eqn, plan)| {
            let KernelChoice::Generated(KernelRequest::Scan(spec)) = program.kernel_choice(eqn)
            else {
                panic!(
                    "a scan equation is not generated: {:?}",
                    program.kernel_choice(eqn)
                );
            };
            let Plan::Dispatches(dispatches) = plan else {
                panic!("a scan launches: {plan:?}");
            };
            (spec.clone(), dispatches.len())
        })
        .collect()
}

/// `qwen4exp.prefill` (GDN), `qwen3next.prefill` (GDN) and `nemotron_h.prefill` (SSD) each plan their scans as
/// generated, admitted schedules on every target, with declared resources (the LDS the body states and the serial
/// work). The bound-driven split of the same requests is `the_work_bound_splits_the_sequence_or_refuses_before_dispatch`
/// (a whole-model graph cannot take a synthetic budget: every other kernel in it is bounded by it too).
#[test]
fn the_model_prefills_plan_bounded_generated_scans() {
    let graphs: [(&str, ScanKernel, Graph<ValidationOutputs>); 3] = [
        (
            "qwen4exp.prefill",
            ScanKernel::DeltaRule,
            registered("qwen4exp", Phase::Prefill, 1, PROMPT, PROMPT),
        ),
        (
            "qwen3next.prefill",
            ScanKernel::DeltaRule,
            registered("qwen3next", Phase::Prefill, 1, PROMPT, PROMPT),
        ),
        (
            "nemotron_h.prefill",
            ScanKernel::Ssd,
            registered("nemotron_h", Phase::Prefill, 1, PROMPT, PROMPT),
        ),
    ];
    for (name, kernel, graph) in &graphs {
        for (target_name, target) in targets() {
            let label = format!("{name} on {target_name}");
            let program = compile(
                graph,
                &target,
                &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
            )
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
            let planned = scans(&program);
            assert!(!planned.is_empty(), "{label}: the prefill plans no scan");
            for (spec, launches) in &planned {
                assert_eq!(spec.kernel, *kernel, "{label}");
                assert_eq!(
                    *launches, 1,
                    "{label}: an unbounded device launches the scan once"
                );
                assert!(
                    spec.lds_bytes() <= u64::from(target.caps.lds_bytes),
                    "{label}: LDS"
                );
                let work = KernelRequest::Scan(spec.clone())
                    .requirements()
                    .serial_work
                    .expect("a scan declares its serial work");
                assert!(work.total() > 0, "{label}");
            }
        }
    }
}
