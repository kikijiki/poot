//! The recurrence scan composite (POOT-634) through compile, for both families: a generated schedule whose
//! body does not grow with the sequence, one launch for the whole sequence, distinct buffers for
//! `y` and `s_out`, and `s_out` published as state. No device.

use poot_graph_ir::test_support::scan::ScanFixture;
use poot_graph_ir::{Graph, OpKind};
use poot_graph_plan::test_support::{gpu_caps, gpu_caps_mut};
use poot_graph_plan::{
    Capability, CompileError, CompileLimits, CompileOptions, KernelChoice, Plan, PlanError,
    Program, Target, compile,
};
use poot_kernelgen::{KernelRequest, ScanKernel, ScanSpec};
use poot_models::model::{LogitRows, StepSpec};
use poot_models::registry::Registry;
use poot_models::test_support::dense::step;
use poot_target::{AmdArch, Backend};

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
            Target::Gpu {
                backend,
                caps: poot_test_util::device_caps::default_caps_for(backend),
            },
        )
    })
}

fn compiled(graph: &Graph, target: &Target) -> Result<Program, CompileError> {
    compile(graph, target, &CompileOptions::new(CompileLimits::STANDARD))
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
            d.body()
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
        let equations: Vec<usize> = lens.map(|len| make(len).graph().graph.eqns().len()).into();
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

/// `y` and `s_out` are distinct values with distinct buffers, both distinct from the packed result they
/// project, and `s_out` is the committed state.
#[test]
fn results_have_distinct_buffers_and_s_out_is_committed() {
    for fixture in families(6) {
        for (name, target) in targets() {
            let program = compiled(&fixture.graph().graph, &target).unwrap();
            let graph = program.graph();
            let packed = graph
                .eqns()
                .iter()
                .find(|e| matches!(e.op, OpKind::Scan(_)))
                .unwrap()
                .out;
            let y = graph.result();
            let (s_in, s_out) = graph.state().next().unwrap();
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
    gpu_caps_mut(&mut target).lds_bytes = 64; // 16 floats: a 32-row column does not fit
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

/// A scan runs as one launch that covers every token, on every backend.
#[test]
fn a_scan_is_one_launch_over_the_whole_sequence() {
    for fixture in families(48) {
        for (name, target) in targets() {
            let label = format!("{:?} {name}", fixture.family);
            let (spec, launches) = scan_plan(&compiled(&fixture.graph().graph, &target).unwrap());
            assert_eq!(launches.len(), 1, "{label}: one launch");
            assert_eq!(spec.len, 48, "{label}: the launch covers the sequence");
            assert_eq!(
                spec.kernel,
                match fixture.family {
                    poot_graph_ir::ScanFamily::DeltaRule => ScanKernel::DeltaRule,
                    poot_graph_ir::ScanFamily::Ssd => ScanKernel::Ssd,
                }
            );
        }
    }
}

/// Small and large per-head states in three regimes: a short sequence keeps one lane per state column; a long
/// sequence whose unsplit launch leaves the device under-filled shares each column's rows between lanes (a
/// state of 16 rows or more); a long sequence with enough heads to fill the device keeps one lane per column.
/// Whatever the regime, the body it plans uses exactly the workgroup memory the request declares (measured from
/// the body, so a declaration cannot understate it), never more than the device offers, and the workgroup is
/// the widest whose columns fit.
#[test]
fn per_head_states_stay_within_the_declared_resources() {
    for (name, target) in targets() {
        let fill = |width: usize| gpu_caps(&target).compute_units as usize * 64 / width + 1;
        for rows in [4usize, 32, 128, 256] {
            for (regime, filled, len, shares) in [
                ("short", false, 16usize, false),
                ("long", false, 1024, true),
                ("filled", true, 1024, false),
            ] {
                // A filled device: enough heads that one lane per column already launches a wave per unit.
                let heads = |width: usize| if filled { fill(width) } else { 2 };
                for fixture in [
                    ScanFixture::delta_rule(heads(rows), len, rows, 8),
                    ScanFixture::ssd(heads(8), len, 8, rows, 8),
                ] {
                    let label = format!("{regime} {:?} rows={rows} on {name}", fixture.family);
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
                    let used = poot_kernelgen::BodyUse::measure(dispatches[0].body())
                        .expect("measurable workgroup memory");
                    assert_eq!(
                        used.lds_bytes,
                        spec.lds_bytes(),
                        "{label}: declared against measured LDS"
                    );
                    assert!(
                        used.lds_bytes <= u64::from(gpu_caps(&target).lds_bytes),
                        "{label}: LDS {} over the device",
                        used.lds_bytes
                    );
                    assert_eq!(
                        used.workgroup,
                        [spec.cols as u32, 1, 1],
                        "{label}: workgroup"
                    );
                    assert!(
                        spec.cols as u32 <= gpu_caps(&target).max_workgroup_size[0]
                            && spec.cols as u32 <= gpu_caps(&target).max_workgroup_invocations,
                        "{label}: {} lanes over the device's workgroup limits",
                        spec.cols
                    );
                    assert_eq!(
                        rows % spec.row_split,
                        0,
                        "{label}: the split divides the rows"
                    );
                    assert!(
                        spec.row_split == 1 || rows / spec.row_split >= 8,
                        "{label}: split {} leaves a lane under 8 rows",
                        spec.row_split
                    );
                    assert_eq!(
                        spec.row_split > 1,
                        shares && rows >= 16,
                        "{label}: row split {} in this regime",
                        spec.row_split
                    );
                    let planes = if spec.row_split == 1 {
                        0
                    } else {
                        2 * spec.row_split
                    };
                    let wider = spec.state_cols() * 2;
                    let lanes_limit = if spec.row_split == 1 { 64 } else { 256 }
                        .min(gpu_caps(&target).max_workgroup_size[0] as usize);
                    assert!(
                        4 * wider * (rows + planes) > gpu_caps(&target).lds_bytes as usize
                            || wider * spec.row_split > lanes_limit
                            || spec.state_cols() == spec.width.min(64),
                        "{label}: {} columns is not the widest that fits",
                        spec.state_cols()
                    );
                }
            }
        }
    }
}

// ---- the model prefills the scan composite is emitted into ------------------------------------------------

const PROMPT: usize = 4;

fn registered(family: &str, rows: usize, tokens: usize, cap: usize) -> Graph {
    let registry = Registry::builtin().expect("the builtin registry");
    // `family/variant` names one of the family's option fixtures (llama's Mixtral, granite's granitemoe).
    let (name, variant) = family.split_once('/').unwrap_or((family, ""));
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == name)
        .unwrap_or_else(|| panic!("no registered family {name}"));
    let fixture = match variant {
        "" => poot_models::test_support::family_fixtures::base(entry.family),
        variant => poot_models::test_support::family_fixtures::variant(entry.family, variant),
    };
    let model =
        (entry.build)(&fixture.raw(), &fixture.store).unwrap_or_else(|e| panic!("{family}: {e}"));
    model
        .model
        .trace(
            &StepSpec::step(step(rows, tokens, cap, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap_or_else(|e| panic!("{family}: {e}"))
}

/// The scans of `program` and, for each, the generated request and the dispatches it launches.
fn scans(program: &Program) -> Vec<(ScanSpec, usize)> {
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
    let graphs: [(&str, ScanKernel, Graph); 3] = [
        (
            "qwen4exp.prefill",
            ScanKernel::DeltaRule,
            registered("qwen4exp", 1, PROMPT, PROMPT),
        ),
        (
            "qwen3next.prefill",
            ScanKernel::DeltaRule,
            registered("qwen3next", 1, PROMPT, PROMPT),
        ),
        (
            "nemotron_h.prefill",
            ScanKernel::Ssd,
            registered("nemotron_h", 1, PROMPT, PROMPT),
        ),
    ];
    for (name, kernel, graph) in &graphs {
        for (target_name, target) in targets() {
            let label = format!("{name} on {target_name}");
            let program = compile(
                graph,
                &target,
                &CompileOptions::new(CompileLimits::STANDARD),
            )
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
            let planned = scans(&program);
            assert!(!planned.is_empty(), "{label}: the prefill plans no scan");
            for (spec, launches) in &planned {
                assert_eq!(spec.kernel, *kernel, "{label}");
                assert_eq!(*launches, 1, "{label}: the scan launches once");
                assert!(
                    spec.lds_bytes() <= u64::from(gpu_caps(&target).lds_bytes),
                    "{label}: LDS"
                );
            }
        }
    }
}

/// The schedule is a function of the scan's layout and the device, not of the model it came from: every scan
/// the registered model prefills plan (at their largest traceable prompt) gets exactly the request a synthetic
/// fixture of the same layout gets, on every target. Renaming a family or adding one cannot change the choice
/// because nothing the planner reads names it. The test configs top out at 64 tokens, under the sequence gate
/// of the shared-column schedule, so the split regimes are pinned by the fixture test above, not here.
#[test]
fn a_model_prefill_scan_gets_the_schedule_of_its_layout() {
    const CAPACITY: usize = 64;
    for (family, kernel) in [
        ("qwen4exp", ScanKernel::DeltaRule),
        ("qwen3next", ScanKernel::DeltaRule),
        ("nemotron_h", ScanKernel::Ssd),
    ] {
        let graph = registered(family, 1, CAPACITY, CAPACITY);
        for (target_name, target) in targets() {
            let label = format!("{family} on {target_name}");
            let program = compile(
                &graph,
                &target,
                &CompileOptions::new(CompileLimits::STANDARD),
            )
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
            let planned = scans(&program);
            assert!(!planned.is_empty(), "{label}: the prefill plans no scan");
            for (spec, _) in planned {
                let fixture = match kernel {
                    ScanKernel::DeltaRule => {
                        ScanFixture::delta_rule(spec.heads, spec.len, spec.width, 8)
                    }
                    ScanKernel::Ssd => {
                        ScanFixture::ssd(spec.heads, spec.len, spec.width, spec.rows, 8)
                    }
                };
                let synthetic = scan_plan(&compiled(&fixture.graph().graph, &target).unwrap()).0;
                assert_eq!(
                    (synthetic.cols, synthetic.row_split),
                    (spec.cols, spec.row_split),
                    "{label}: the model's scan and a synthetic scan of the same layout differ"
                );
            }
        }
    }
}

/// The row split of the one scan `fixture` plans on `target`.
fn planned_row_split(fixture: &ScanFixture, target: &Target) -> usize {
    scan_plan(&compiled(&fixture.graph().graph, target).unwrap())
        .0
        .row_split
}

/// SSD shares the rows in the same regime as the delta rule (long sequence, device under-filled, state of
/// 128 rows): its split was measured to win there on wgpu and ROCm (POOT-1112 bank cases), so the family does
/// not enter the rule. Mutation: in `scan_geometry`, return the unsplit shape when `layout.state_rows` is
/// not a delta rule's `layout.width` (a family test by another name); the SSD row goes red with `row_split 1`.
#[test]
fn ssd_splits_in_the_same_regime_as_the_delta_rule() {
    for (name, target) in targets() {
        let delta = ScanFixture::delta_rule(2, 1024, 128, 8);
        let ssd = ScanFixture::ssd(2, 1024, 8, 128, 8);
        assert!(
            planned_row_split(&delta, &target) > 1,
            "{name}: the delta rule is split in the long, under-filled regime"
        );
        assert!(
            planned_row_split(&ssd, &target) > 1,
            "{name}: SSD in the same regime shares the rows"
        );
    }
}

/// The sequence gate is exactly 512 steps: 511 keeps one lane per column, 512 shares the rows (the device
/// is under-filled by two heads, the delta rule). Mutation: change `MIN_SPLIT_STEPS` to 511 or 513; the 511
/// row (`row_split` 16 where 1 is due) or the 512 row goes red.
#[test]
fn the_sequence_gate_sits_at_512_steps() {
    for (name, target) in targets() {
        let at = |len| planned_row_split(&ScanFixture::delta_rule(2, len, 128, 8), &target);
        assert_eq!(at(511), 1, "{name}: 511 steps");
        assert!(at(512) > 1, "{name}: 512 steps share the rows");
    }
}
