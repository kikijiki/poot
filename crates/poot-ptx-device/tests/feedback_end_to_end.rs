//! POOT-1505 / POOT-1509: the engine plans the Flash-Next expert projection (packed Iq2_Xs, `K` 2560, `N` 640, 512
//! experts) with the PTX device's compile feedback, at the routed rows of a 128-, 2048- and 8192-token prompt (ten
//! experts a token), under the default narrowed precision and under the exact one.
//!
//! Before the feedback the narrowed plan's fragment tiling staged a whole round of packed weight loads ahead of its
//! multiply, the compiler spilled them into local memory inside the `K` loop and the launch took 35 ms where the
//! exact one took 2.4 (POOT-1509). With it the selection compiles each candidate in cost order, takes the first the
//! driver does not spill, and keeps the exact tiling unless the narrowed one is cheaper. The row holds the default
//! plan to the exact plan's device time, and every planned contraction to a driver verdict with no local memory. A
//! host without an NVIDIA device skips.

use std::sync::Arc;

use poot_executor::{Device, DeviceTime, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_graph_ir::test_support::packed_graphs::{
    PackedExpertStack, TEST_STACK, packed_indexed_linear,
};
use poot_graph_ir::{Builder, IndexRange, MatmulPrecision, Slot, TensorType};
use poot_graph_plan::staged::{
    DeviceId, DevicePlacement, ExpertPlacement, Partition, TargetSet, compile_staged,
};
use poot_graph_plan::{CompileLimits, RegisterProbe};
use poot_ptx_device::PtxDevice;
use poot_quant::PackedPayload;
use poot_quant::format::WeightFormat;
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_target::KernelCompiler;
use poot_tensor::{DType, HostTensor, HostView};

const K: usize = 2560;
const N: usize = 640;
const EXPERTS: usize = 512;
const FORMAT: WeightFormat = WeightFormat::Iq2_Xs;
const WARMUP: usize = 3;
const TIMED: usize = 7;
/// The routed rows of a 128-, 2048- and 8192-token prompt, ten experts a token.
const ROWS: [usize; 3] = [1280, 20480, 81920];

/// How many times the exact plan's device time the default plan may take: the owner's requirement is that the default
/// precision is no slower than exact; the rest is the run-to-run noise of one device span.
const NO_SLOWER: f64 = 1.10;
/// The narrowed output's bound against the exact one, relative to the exact output's largest magnitude (f16 operands
/// summed over 2560 terms; the same bound as `packed_grouped_narrow`).
const NARROW_TOLERANCE: f64 = 2e-3;

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// What one plan did: its output, median device span in microseconds, and the contraction it planned.
struct Run {
    output: Vec<f32>,
    span_us: f64,
    /// `(kernel key, grid, workgroup, registers, local bytes)` of every contraction the plan launches.
    contractions: Vec<String>,
    local_bytes: u32,
    compiler: Option<KernelCompiler>,
}

fn run(gpu: PtxDevice, rows: usize, precision: Option<MatmulPrecision>) -> Run {
    // Eight distinct experts' payloads cycled over the stack: the table has its real stride and the rows read
    // different weights.
    let payloads: Vec<Arc<PackedPayload>> = (0..8)
        .map(|seed| {
            Arc::new(poot_quant::test_support::packed::prescribed_scale_payload(
                FORMAT,
                [N, K],
                31 + seed,
            ))
        })
        .collect();
    let mut builder = WeightStore::builder();
    for expert in 0..EXPERTS {
        builder
            .insert(
                format!("{}.{expert}", TEST_STACK.const_name()),
                WeightEntry::Packed(Arc::clone(&payloads[expert % payloads.len()])),
            )
            .unwrap();
    }
    let store = builder.build();

    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![rows, K]))
        .unwrap();
    let ids = b
        .slot_named_ranged(
            Slot::Activation,
            "moe-ids",
            TensorType::new(vec![rows], DType::I32),
            IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let table = PackedExpertStack {
        stack: TEST_STACK,
        descriptor: payloads[0].weight(),
        experts: EXPERTS,
    };
    let y = packed_indexed_linear(&b, x, ids, &table).unwrap();
    let graph = b.finish(y).unwrap();
    let (key_x, key_ids) = (
        graph.meta(x.id()).slot_key().unwrap().clone(),
        graph.meta(ids.id()).slot_key().unwrap().clone(),
    );
    let activation = HostTensor::f32(
        vec![rows, K],
        (0..rows * K)
            .map(|i| (splitmix(i as u64) % 2000) as f32 / 1000.0 - 1.0)
            .collect(),
    );
    let routed: Vec<i32> = (0..rows)
        .map(|m| (splitmix(1_000_003 + m as u64) % EXPERTS as u64) as i32)
        .collect();
    let routed_bytes: Vec<u8> = routed.iter().flat_map(|v| v.to_le_bytes()).collect();

    let target = Device::target(&gpu);
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(gpu));
    let mut weights = ConstWeights::new(&store);
    let bound = weights.graph(&graph);
    let exe = exec
        .load_weights(Arc::new(store.clone()), Arc::new(weights.map()))
        .unwrap();
    // The engine's own options: the device's feedback included. Only the precision is chosen here.
    let mut options = exec.compile_options(CompileLimits::STANDARD);
    let feedback = options
        .register_feedback
        .clone()
        .expect("the PTX engine plans with the driver's feedback");
    if let Some(precision) = precision {
        options = options.with_matmul_precision(precision);
    }
    let program = compile_staged(
        &bound,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &options,
    )
    .unwrap();

    let mut contractions = Vec::new();
    let mut local_bytes = 0;
    let mut compiler = None;
    for (_, _, stage) in program.stages() {
        compiler = stage.register_feedback();
        for (_, plan) in stage.planned() {
            for d in plan.dispatches() {
                if !d.label().starts_with("contraction") {
                    continue;
                }
                let verdict = feedback
                    .oracle()
                    .compile(&RegisterProbe {
                        body: d.body().clone(),
                    })
                    .expect("the driver compiles a planned contraction");
                local_bytes = local_bytes.max(verdict.local_bytes);
                contractions.push(format!(
                    "{} grid {:?} workgroup {:?}: {} registers, {} B local",
                    d.key(),
                    d.launch().grid,
                    d.body().workgroup_size,
                    verdict.registers,
                    verdict.local_bytes
                ));
            }
        }
    }

    let entry = exec.add_entry(exe, &program).unwrap();
    let (mut output, mut spans) = (Vec::new(), Vec::new());
    let ids_shape = [rows];
    let routed_view = HostView::new(DType::I32, rows, &routed_bytes).unwrap();
    for step in 0..WARMUP + TIMED {
        let mut inputs = StepInputs::new();
        inputs.push(key_x.clone(), activation.shape(), activation.view());
        inputs.push(key_ids.clone(), &ids_shape, routed_view);
        let mut out = exec.step(exe, entry, &inputs, &mut NoSync).unwrap();
        let device_time = out.device_time();
        let DeviceTime::Measured(measured) = &device_time else {
            panic!("the device reports no timestamps");
        };
        let span = measured.device_span.expect("a device span");
        if step >= WARMUP {
            spans.push(span.as_secs_f64() * 1e6);
        }
        if step == 0 {
            output = out.to_host().unwrap().as_f32().unwrap().to_vec();
        }
    }
    spans.sort_by(f64::total_cmp);
    Run {
        output,
        span_us: spans[spans.len() / 2],
        contractions,
        local_bytes,
        compiler,
    }
}

/// The default plan of a routed expert projection takes no longer than the exact plan, agrees with its output, is
/// named for the driver whose verdicts chose it, and launches no kernel the driver compiles with local memory, at the
/// routed rows of a 128-, 2048- and 8192-token prompt. Mutation: return `None` from `PtxDevice::kernel_registers`;
/// the plan is the register model's again, names no compiler, and its narrowed tiling spills.
#[test]
fn the_engine_plans_the_expert_projection_with_the_drivers_verdicts_on_ptx() {
    for rows in ROWS {
        let Some(gpu) = poot_runtime_common::device_skip::open_or_skip(
            DeviceBackend::Ptx,
            PtxDevice::new_with_device_timing(),
        ) else {
            return;
        };
        let identity = gpu
            .kernel_registers()
            .expect("the PTX device offers its compiler")
            .compiler();
        let exact = run(gpu, rows, Some(MatmulPrecision::Exact));
        let gpu = PtxDevice::new_with_device_timing().expect("a second handle on the device");
        let default = run(gpu, rows, None);
        println!(
            "rows {rows}: exact {:.0} us ({}), default {:.0} us ({}), ratio {:.2}",
            exact.span_us,
            exact.contractions.join("; "),
            default.span_us,
            default.contractions.join("; "),
            default.span_us / exact.span_us
        );
        for plan in [&exact, &default] {
            assert_eq!(
                plan.compiler,
                Some(identity),
                "rows {rows}: the plan names its compiler"
            );
            assert_eq!(
                plan.local_bytes, 0,
                "rows {rows}: a planned contraction spills: {:?}",
                plan.contractions
            );
        }
        let scale = exact
            .output
            .iter()
            .fold(0.0f64, |m, e| m.max(f64::from(*e).abs()));
        let worst = default
            .output
            .iter()
            .zip(&exact.output)
            .map(|(n, e)| (f64::from(*n) - f64::from(*e)).abs() / scale)
            .fold(0.0f64, f64::max);
        assert!(
            worst <= NARROW_TOLERANCE,
            "rows {rows}: the default output is {worst:.2e} of the exact output's range from it"
        );
        assert!(
            default.span_us <= NO_SLOWER * exact.span_us,
            "rows {rows}: the default plan takes {:.0} us, {:.2} times the exact plan's {:.0} us",
            default.span_us,
            default.span_us / exact.span_us,
            exact.span_us
        );
    }
}
