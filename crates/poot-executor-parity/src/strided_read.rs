//! POOT-1139: the strided-read ceiling microbenchmark, model-free and the same on every backend.
//!
//! The mechanism under test (research note 2026-10-12, section 2): DRAM channels are selected by address bits
//! [11:8], a 256 B granule over 16 channels, so a read whose row stride is `stride_bytes` reaches
//! `16 / gcd(stride_bytes / 256, 16)` channels (`poot_test_util::dram_channels::alias_factor`). The probe
//! reads exactly one granule from each row of a `[rows, stride / 4]` f32 activation slot (a slice of the
//! first `GRANULE_BYTES / 4` columns, reduced along the row), so the only thing that changes between two
//! measurements is the row stride.
//!
//! [`measure_strides`] runs the graph through any [`Executor`] built with detailed device timing and
//! [`check_order`] asserts the order the channel model implies and [`report`] prints the fit; the device crates' tests call both.

use std::time::Duration;

use poot_executor::{DeviceTime, Executor, MeasuredDeviceTime, NoSync, StepInputs};
use poot_graph_ir::op::RedOp;
use poot_graph_ir::{Builder, Graph, Slot, TensorType};
use poot_graph_plan::Target;
use poot_tensor::HostTensor;
use poot_test_util::dram_channels::{CHANNELS, GRANULE_BYTES, alias_factor};

use crate::scan::load_graph_entry;

/// The channels a read at `row_stride_bytes` per row, one granule per row, reaches:
/// `16 / gcd(stride / 256, 16)`.
fn predicted_channels(row_stride_bytes: usize) -> usize {
    CHANNELS / alias_factor(row_stride_bytes)
}

/// The graph: an `f32 [rows, stride_bytes / 4]` activation slot, its first granule of columns per row, summed
/// along the row to `[rows]`.
fn graph(rows: usize, row_stride_bytes: usize) -> Graph {
    let b = Builder::for_tests();
    let cols = row_stride_bytes / 4;
    let x = b
        .slot_named(
            Slot::Activation,
            "strided",
            TensorType::f32(vec![rows, cols]),
        )
        .expect("the slot");
    let granule = b
        .slice(x, 1, 0, GRANULE_BYTES / 4)
        .expect("the first granule of every row");
    let sum = b.reduce(RedOp::Sum, granule, 1, false).expect("reduce");
    b.finish(sum).expect("the strided-read graph is valid")
}

/// The device time of `runs` steps of the probe at `row_stride_bytes` per row, after one warm-up step.
fn measure(
    exec: &mut dyn Executor,
    target: Target,
    rows: usize,
    row_stride_bytes: usize,
    runs: usize,
) -> Vec<MeasuredDeviceTime> {
    let graph = graph(rows, row_stride_bytes);
    let (slot, _) = graph.slots().first().expect("one slot");
    let key = graph.meta(*slot).slot_key().expect("a slot key").clone();
    let (exe, entry) = load_graph_entry(exec, target, &graph, "strided read");
    let shape = vec![rows, row_stride_bytes / 4];
    let data = HostTensor::f32(shape.clone(), vec![1.0; rows * shape[1]]);
    let mut times = Vec::with_capacity(runs);
    for run in 0..=runs {
        let mut inputs = StepInputs::new();
        inputs.push(key.clone(), &shape, data.view());
        let mut out = exec
            .step(exe, entry, &inputs, &mut NoSync)
            .expect("the strided-read step");
        let DeviceTime::Measured(measured) = out.device_time() else {
            panic!("the strided-read probe needs a device built with device timing");
        };
        out.to_host().expect("the sums read back");
        if run > 0 {
            times.push(measured);
        }
    }
    exec.unload(exe).unwrap();
    times
}

/// The strided-read dispatch of one timed step: the longest of its dispatches. The probe plans as the
/// strided read of the granules followed by a small contiguous reduce, and the read is the long one at every
/// stride the audit uses (a step with no detailed dispatch list is an error: the device was not built with
/// detailed device timing).
fn read_time(measured: &MeasuredDeviceTime) -> Duration {
    measured
        .dispatches
        .iter()
        .map(|dispatch| dispatch.duration)
        .max()
        .expect("a timed step lists its dispatches")
}

/// One stride's measurement: `runs` read rates, in GB/s of granule bytes read per second of device time.
#[derive(Clone, Debug)]
pub struct StrideRates {
    /// Byte distance between consecutive granule reads.
    pub stride_bytes: usize,
    /// The channels this stride reaches at the probe's one granule per row.
    pub channels: usize,
    /// Read rate of each run, in GB/s, in run order.
    pub gbps: Vec<f64>,
}

impl StrideRates {
    /// The middle run's rate (the upper-middle for an even count), in GB/s.
    pub fn median(&self) -> f64 {
        let mut sorted = self.gbps.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    }

    /// The fastest run: contention from another process on the shared device only lowers a run's rate, so the
    /// peak is the estimate that survives it (the median does not: a run-5 hold on wgpu read 7.4 GB/s at the
    /// unaliased stride while a second process held the GPU, with a peak of the usual 50).
    pub fn peak(&self) -> f64 {
        self.gbps.iter().copied().fold(0.0, f64::max)
    }

    /// `(max - min) / median` over the runs: the run-to-run noise this stride was measured under.
    pub fn spread(&self) -> f64 {
        let (lo, hi) = self
            .gbps
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        (hi - lo) / self.median()
    }
}

/// The read rate at each of `strides` bytes per row: `runs` timed steps each, after one warm-up step, over
/// `rows` rows of one granule. The caller's device must have been built with detailed device timing.
pub fn measure_strides(
    exec: &mut dyn Executor,
    target: Target,
    rows: usize,
    strides: &[usize],
    runs: usize,
) -> Vec<StrideRates> {
    let read_bytes = (rows * GRANULE_BYTES) as f64;
    strides
        .iter()
        .map(|&stride_bytes| {
            let gbps = measure(exec, target, rows, stride_bytes, runs)
                .iter()
                .map(|measured| read_bytes / read_time(measured).as_secs_f64() / 1e9)
                .collect();
            StrideRates {
                stride_bytes,
                channels: predicted_channels(stride_bytes),
                gbps,
            }
        })
        .collect()
}

/// The smallest ratio between the peak read rate of a stride that reaches more channels and one that reaches
/// fewer, per step of the order the probe asserts: 2 over 1 channel, 4 over 2, and all 16 over 1. Each is below
/// the smallest per-run ratio of five clean holds on both backends (2 over 1: 1.68; 4 over 2: 1.44; 16 over 1:
/// 3.09), so the order is the robust part of the channel model. The fitted rates are printed by [`report`],
/// never asserted: the box is shared with a model server, and a hold that overlapped another GPU user read 7.4
/// GB/s where 50 is usual (the peak over nine runs ignores such a run; a median does not).
pub const ORDER_FLOORS: [(usize, usize, f64); 3] = [(1, 2, 1.3), (2, 4, 1.2), (1, 16, 2.0)];

fn median_at(rates: &[StrideRates], channels: usize) -> &StrideRates {
    rates
        .iter()
        .find(|r| r.channels == channels)
        .unwrap_or_else(|| panic!("the probe measures a stride that reaches {channels} channels"))
}

/// The order the channel model implies, in terms that survive noise: a stride that reaches more channels peaks
/// at least [`ORDER_FLOORS`] times faster than one that reaches fewer. `Err` names each broken step.
pub fn check_order(rates: &[StrideRates]) -> Result<(), String> {
    let broken: Vec<String> = ORDER_FLOORS
        .iter()
        .filter_map(|&(few, many, floor)| {
            let (slow, fast) = (median_at(rates, few), median_at(rates, many));
            let ratio = fast.peak() / slow.peak();
            (ratio < floor).then(|| {
                format!(
                    "{} B ({many} channels) peaks at {:.1} GB/s against {:.1} GB/s at {} B ({few} channels): \
                     ratio {ratio:.2}, floor {floor:.1}",
                    fast.stride_bytes,
                    fast.peak(),
                    slow.peak(),
                    slow.stride_bytes
                )
            })
        })
        .collect();
    if broken.is_empty() {
        Ok(())
    } else {
        Err(broken.join("\n"))
    }
}

/// The measured table with the channel model beside it: `min(c * r, ceiling)` with `r` the one-channel and
/// `ceiling` the all-channel median. A report; nothing in it is asserted.
pub fn report(rates: &[StrideRates]) -> String {
    let per_channel = median_at(rates, 1).median();
    let ceiling = median_at(rates, CHANNELS).median();
    let mut out =
        String::from("stride B  channels  median GB/s  peak GB/s  model GB/s  error  spread\n");
    for r in rates {
        let model = (r.channels as f64 * per_channel).min(ceiling);
        out.push_str(&format!(
            "{:>8}  {:>8}  {:>11.1}  {:>9.1}  {:>10.1}  {:>+5.0}%  {:>6.2}\n",
            r.stride_bytes,
            r.channels,
            r.median(),
            r.peak(),
            model,
            (r.median() / model - 1.0) * 100.0,
            r.spread()
        ));
    }
    out
}
