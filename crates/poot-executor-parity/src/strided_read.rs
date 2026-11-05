//! POOT-1139: the strided-read ceiling microbenchmark, model-free and the same on every backend.
//!
//! The mechanism under test (research note 2026-10-12, section 2): DRAM channels are selected by address bits
//! [11:8], a 256 B granule over 16 channels, so a read whose row stride is `stride_bytes` reaches
//! `16 / gcd(stride_bytes / 256, 16)` channels (`poot_target::dram_channels::alias_factor`). The probe
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
use poot_target::dram_channels::{CHANNELS, GRANULE_BYTES, alias_factor};
use poot_tensor::HostTensor;

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
    let (exe, entry, _) = load_graph_entry(exec, target, &graph, "strided read");
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

    /// The fastest run. A report column only: it is an extreme of the sample, so one run whose timestamps span
    /// far less than the dispatch (an outlier of 115 GB/s at the one-channel stride, above the 105 GB/s the
    /// contiguous read reaches, under CPU load) moves it arbitrarily, and the order check does not use it.
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

/// The smallest ratio between the median read rate of a stride that reaches more channels and one that reaches
/// fewer, per step of the order the probe asserts: 2 over 1 channel, 4 over 2, and all 16 over 1. Each is below
/// the smallest ratio of the recorded medians on both backends (2 over 1: 1.78; 4 over 2: 1.84; 16 over 1:
/// 8.7), so the order is the robust part of the channel model. The fitted rates are printed by [`report`],
/// never asserted. The median of nine runs ignores up to four runs that read wrongly in either direction (a
/// run slowed by another GPU user, a run whose timestamps span far less than its dispatch); the peak or the
/// minimum of the sample takes one such run as the estimate.
pub const ORDER_FLOORS: [(usize, usize, f64); 3] = [(1, 2, 1.3), (2, 4, 1.2), (1, 16, 2.0)];

fn median_at(rates: &[StrideRates], channels: usize) -> &StrideRates {
    rates
        .iter()
        .find(|r| r.channels == channels)
        .unwrap_or_else(|| panic!("the probe measures a stride that reaches {channels} channels"))
}

/// The order the channel model implies, in terms that survive noise: a stride that reaches more channels reads
/// at least [`ORDER_FLOORS`] times faster (median rate) than one that reaches fewer. `Err` names each broken step.
pub fn check_order(rates: &[StrideRates]) -> Result<(), String> {
    let broken: Vec<String> = ORDER_FLOORS
        .iter()
        .filter_map(|&(few, many, floor)| {
            let (slow, fast) = (median_at(rates, few), median_at(rates, many));
            let ratio = fast.median() / slow.median();
            (ratio < floor).then(|| {
                format!(
                    "{} B ({many} channels) reads a median {:.1} GB/s against {:.1} GB/s at {} B ({few} channels): \
                     ratio {ratio:.2}, floor {floor:.1}",
                    fast.stride_bytes,
                    fast.median(),
                    slow.median(),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Rates for the strides the order check reads: the median of `runs` per stride, nothing else.
    fn rates(channels: usize, gbps: Vec<f64>) -> StrideRates {
        StrideRates {
            stride_bytes: 4096 * channels,
            channels,
            gbps,
        }
    }

    fn model_rates(one_channel: Vec<f64>) -> Vec<StrideRates> {
        vec![
            rates(1, one_channel),
            rates(2, vec![21.5; 9]),
            rates(4, vec![39.6; 9]),
            rates(16, vec![104.9; 9]),
        ]
    }

    /// A single run whose timestamps span far less than its dispatch (115 GB/s at the one-channel stride, seen
    /// under CPU load on wgpu, above the contiguous ceiling) must not decide the order. Mutation: take
    /// `peak()` in `check_order`; the one-channel stride then reads 115 GB/s and this goes red on
    /// "ratio 0.19, floor 1.3".
    #[test]
    fn one_outlier_run_does_not_decide_the_order() {
        let mut one = vec![12.1; 9];
        one[4] = 115.0;
        check_order(&model_rates(one)).expect("eight clean runs of nine hold the order");
    }

    /// The other side: when the one-channel stride reads as fast as the others in every run, the order is
    /// broken and `check_order` names it.
    #[test]
    fn a_stride_set_with_no_channel_effect_is_refused() {
        let flat = vec![
            rates(1, vec![40.0; 9]),
            rates(2, vec![40.0; 9]),
            rates(4, vec![40.0; 9]),
            rates(16, vec![40.0; 9]),
        ];
        let broken = check_order(&flat).expect_err("a flat read rate breaks the channel order");
        assert!(
            broken.contains("floor 1.3"),
            "names the broken step: {broken}"
        );
    }
}
