//! POOT-1072: the profiled executor the bench runner's `--profile` opens (`open_timed_executor`) reports the
//! device's own time through the driver: a nonzero sum for a generation on wgpu (timestamp queries), the
//! nonzero sum on ROCm (HSA dispatch timestamps), and the typed `Unknown` on the untimed `open_executor`. The window is read
//! exactly as the runner reads it: a mark before the generation, then `Report::window` over the entries
//! that stepped since.

#[path = "add_a_model/family.rs"]
mod family;

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use poot_executor::DetailedTiming;
use poot_graph_plan::{CompileLimits, CompileOptions, Submission};
use poot_llm::driver::{
    BackendChoice, DraftPolicy, Driver, DriverOptions, GenerateRequest, Head, Layout, LoadOptions,
    ModelHandle, PreparedSetLimits, ServingShapes, Warm, open_executor, open_timed_executor,
    program_retention,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::registry::{Fixture, Registry};
use poot_profile::{Report, WindowReport};
use poot_runtime_common::DeviceBackend;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

const CAPACITY: usize = 64;
const NEW_TOKENS: usize = 16;

/// The bounded detail every timed row retains: every step, within finite record and query bounds.
fn detail() -> DetailedTiming {
    DetailedTiming::default_diagnostic()
}

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn tokenizer() -> Tokenizer {
    let mut words: Vec<String> = ('a'..='z').map(String::from).collect();
    words.extend(('0'..='9').map(String::from));
    words.extend((0..family::VOCAB - words.len()).map(|i| format!("p{i}")));
    let vocab = words
        .into_iter()
        .enumerate()
        .map(|(i, w)| (w, i as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("a".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
}

fn write_checkpoint(dir: &Path, fixture: &Fixture) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), fixture.config.to_string()).unwrap();
    tokenizer().save(dir.join("tokenizer.json"), false).unwrap();
    poot_test_util::safetensors::write(&dir.join("model.safetensors"), &fixture.store);
}

fn options() -> DriverOptions {
    let compile = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(CAPACITY),
        prefill_chunk: nz(8),
        max_trace_tokens: nz(CAPACITY),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
    }
}

/// Generate [`NEW_TOKENS`] tokens on a fixture checkpoint through `executor`: the tokens and the timing
/// window of that generation.
fn profiled_generation(executor: Box<dyn poot_executor::Executor>) -> (Vec<u32>, WindowReport) {
    let dir = poot_test_util::unique_temp_path("timed_executor");
    write_checkpoint(&dir, &family::fixture());
    let mut registry = Registry::builtin().unwrap();
    registry.register(family::ENTRY).unwrap();
    let handle = Arc::new(ModelHandle::load(&dir, &registry, &LoadOptions::default()).unwrap());
    let mut driver = Driver::new(handle, executor, options()).unwrap();
    let prompt = vec![1, 2, 3, 4];
    driver
        .prepare(&ServingShapes {
            layout: Layout::Contiguous,
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY],
            warm: Warm::Prompts(&[prompt.len()]),
            windows: &[],
            adapters: &[],
        })
        .unwrap();
    let mark = driver.executor_stats().timing.mark();
    let generated = driver
        .generate(
            GenerateRequest {
                prompt,
                max_new: NEW_TOKENS,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: true,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .unwrap();
    let timing = driver.executor_stats().timing;
    let entries = timing.entries_with_activity_since(&mark);
    assert!(!entries.is_empty(), "the generation stepped no entry");
    (generated.tokens, Report::window(&timing, &mark, &entries))
}

fn profiled_window(executor: Box<dyn poot_executor::Executor>) -> WindowReport {
    profiled_generation(executor).1
}

/// A profiled wgpu executor measures the generation's device time. Mutation: `open_timed_executor`'s wgpu arm
/// opens `WgpuDevice::new()` instead of `new_with_device_timing`; the window carries no device sum and the
/// row goes red.
#[test]
fn profiled_executor_reports_a_nonzero_device_time_wgpu() {
    let Some(executor) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Wgpu,
        open_timed_executor(BackendChoice::Wgpu, detail()),
    ) else {
        return;
    };
    let window = profiled_window(executor);
    let sum = window
        .device
        .sum_of_dispatch_durations
        .expect("a timed wgpu executor measures a device sum");
    assert!(!sum.is_zero(), "the measured device sum is zero");
}

/// A profiled ROCm executor measures the generation's device time from the HSA dispatch timestamps.
/// Mutation: `open_timed_executor`'s ROCm arm opens `RocmDevice::new()` instead of
/// `new_with_device_timing`; the window carries no device sum and the row goes red.
#[test]
fn profiled_executor_reports_a_nonzero_device_time_rocm() {
    let Some(executor) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Rocm,
        open_timed_executor(BackendChoice::Rocm, detail()),
    ) else {
        return;
    };
    let window = profiled_window(executor);
    let sum = window
        .device
        .sum_of_dispatch_durations
        .expect("a timed rocm executor measures a device sum");
    assert!(!sum.is_zero(), "the measured device sum is zero");
}

/// The untimed executor never fabricates a device time. Mutation: have `open_executor` open the timed wgpu
/// device; the window then carries a sum and this row goes red.
#[test]
fn unprofiled_executor_reports_unknown_device_time_wgpu() {
    let Some(executor) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Wgpu,
        open_executor(BackendChoice::Wgpu),
    ) else {
        return;
    };
    let window = profiled_window(executor);
    assert_eq!(window.device.sum_of_dispatch_durations, None);
}

/// Timing must not change what is generated: the same prompt, greedy, yields the same token ids through the
/// timed executor (one pass per dispatch, timestamps resolved) as through the untimed one. Mutation: have
/// `WgpuDevice::replay` return early when the context times device work; the timed run replays nothing and
/// its first sample fails (or its ids differ).
#[test]
fn a_profiled_generation_equals_the_unprofiled_one_wgpu() {
    let Some(timed) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Wgpu,
        open_timed_executor(BackendChoice::Wgpu, detail()),
    ) else {
        return;
    };
    let Some(untimed) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Wgpu,
        open_executor(BackendChoice::Wgpu),
    ) else {
        return;
    };
    let (timed_tokens, _) = profiled_generation(timed);
    let (untimed_tokens, _) = profiled_generation(untimed);
    assert_eq!(timed_tokens.len(), NEW_TOKENS);
    assert_eq!(timed_tokens, untimed_tokens);
}

/// Timing must not change what ROCm generates: the same prompt, greedy, yields the same token ids through
/// the timed executor (a signal per dispatch, profiler on) as through the untimed one. Mutation: have the
/// timed `RocmContext::enqueue` drop the final dispatch from the published batch; the timed run's ids differ.
#[test]
fn a_profiled_generation_equals_the_unprofiled_one_rocm() {
    let Some(timed) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Rocm,
        open_timed_executor(BackendChoice::Rocm, detail()),
    ) else {
        return;
    };
    let Some(untimed) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Rocm,
        open_executor(BackendChoice::Rocm),
    ) else {
        return;
    };
    let (timed_tokens, _) = profiled_generation(timed);
    let (untimed_tokens, _) = profiled_generation(untimed);
    assert_eq!(timed_tokens.len(), NEW_TOKENS);
    assert_eq!(timed_tokens, untimed_tokens);
}

/// The PTX arm of `open_timed_executor` brackets each replay with a CUDA event pair: the window carries a
/// nonzero device span (PTX times the step, not each dispatch, so no per-dispatch sum). Mutation: open
/// `PtxDevice::new()` in the timed arm; the span is absent and the row goes red.
#[test]
fn profiled_executor_reports_a_nonzero_device_span_ptx() {
    let Some(executor) = poot_test_util::device_skip::open_or_skip(
        DeviceBackend::Ptx,
        open_timed_executor(BackendChoice::Ptx, detail()),
    ) else {
        return;
    };
    let window = profiled_window(executor);
    let span = window
        .device
        .device_span
        .expect("a timed ptx executor measures a device span");
    assert!(!span.is_zero(), "the measured device span is zero");
}
