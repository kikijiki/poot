//! POOT-1438: the tier-3 gates of the narrow operand path (`receipts/1429-narrow-operands/design.md` 6.3).
//!
//! A real checkpoint is compiled twice on one device and one weight upload: once under
//! [`MatmulPrecision::Exact`] and once under [`MatmulPrecision::Narrow`]. Both programs score the same token
//! windows teacher-forced, with the logits at every position, and the two next-token distributions are compared
//! position by position:
//!
//! - **mean KL**: the mean of KL(exact || narrow) in nats, gated at 1e-3;
//! - **max shift**: the largest `|logit_narrow - logit_exact|` over every logit of the gate text, gated at a
//!   ceiling recorded per row ([`Gate`]);
//! - **top-1 agreement** and the **flips** (positions where the two programs rank a different token first) with
//!   the largest exact top-two gap at a flip, recorded for the report.
//!
//! The greedy criterion is ADR-0101 decision 3's tie rule, not a top-1 fraction: a flip is allowed only where the
//! reference's (Exact's) gap between its top token and the narrow program's top token is within the tie
//! tolerance delta = 2 x the recorded max-shift ceiling (`receipts/1454-narrow-dtype-policy/design.md`
//! section 9). The max-shift gate enforces it: a flip at position p means
//! `narrow[t_n] >= narrow[t_e]`, so `exact[t_e] - exact[t_n] <= shift(t_n) - shift(t_e) <= 2 x max shift`; a
//! program within its shift ceiling therefore cannot flip a token beyond the tie tolerance, and one that does
//! exceeds the ceiling. The ceiling is recorded, so a broken narrowing cannot widen its own tolerance.
//!
//! This is llama.cpp's `llama-perplexity --kl-divergence` method with the exact program as the reference, and
//! the method POOT-1395 used for the f16 KV cache. A row also checks that the narrow program really narrowed
//! (it holds narrowed ops of the expected dtype) and that the exact program did
//! not, so a device without a matrix unit cannot pass by comparing two exact programs.
//!
//! Small models only; a 20B+ checkpoint never runs locally.

use std::collections::BTreeMap;
use std::path::Path;

use poot_graph_ir::analysis::Tier2Class;
use poot_graph_ir::{FragmentDtype, MatmulPrecision, OperandPrecision};
use poot_graph_plan::{CompileLimits, CompileOptions};
use poot_models::registry::Registry;
use poot_runtime_common::DeviceBackend;

use super::llamacpp_kl_tests::{CORPUS, TeacherForced};
use super::{BackendChoice, ModelHandle, open_executor};

/// Tokens per scored window.
const N_CTX: usize = 128;

/// Windows scored: 640 positions in all.
const WINDOWS: usize = 5;

/// How closely the narrow program's next-token distributions follow the exact program's.
#[derive(Clone, Debug, PartialEq)]
struct Agreement {
    positions: usize,
    /// Mean of KL(exact || narrow), in nats.
    mean_kl: f64,
    /// Largest KL at one position.
    max_kl: f64,
    /// Fraction of positions where both programs rank the same token first.
    top1: f64,
    /// Largest `|logit_narrow - logit_exact|` over every logit of every position.
    max_abs_logit: f64,
    /// Positions where the two programs rank a different token first.
    flips: usize,
    /// Largest `exact[t_exact] - exact[t_narrow]` over the flips: how far from a tie the reference was where the
    /// narrow program flipped (0 when nothing flipped).
    max_flip_margin: f64,
}

/// The mean-KL ceiling every row meets: the design's (1429 section 6.3), not loosened.
const MAX_MEAN_KL: f64 = 1e-3;

/// What a row's [`Agreement`] must meet.
///
/// `max_shift` is the row's recorded ceiling on the largest logit shift: the shift its narrow program measured
/// (POOT-1454, Strix Halo) times [`SHIFT_MARGIN`], rounded up to two significant figures. The tie tolerance of
/// ADR-0101 decision 3 is `2 x max_shift`, and a program within the ceiling meets it (module docs).
#[derive(Clone, Copy, Debug)]
struct Gate {
    max_mean_kl: f64,
    max_shift: f64,
}

/// The margin of a recorded shift ceiling over the measured shift. A row is deterministic on its device (the
/// POOT-1454 re-run reproduced POOT-1438's numbers to every printed digit), so the margin is not for noise: it
/// covers a different but equally correct schedule of the same narrow semantics. Its measure is the spread of the
/// f16 rows across ROCm, wgpu and raw Vulkan, which differ only in schedule: the largest ratio of one checkpoint's
/// shifts across the three is 2.35 (Qwen3 0.6B, 0.0599 / 0.0255), rounded up to 2.4. bf16 has one backend on
/// the box and takes the same margin.
const SHIFT_MARGIN: f64 = 2.4;

/// `x > 0` rounded up to two significant figures (0.49104 is 0.50, 5.5562 is 5.6).
fn round_up_to_two_figures(x: f64) -> f64 {
    let unit = 10f64.powf(x.log10().floor() - 1.0);
    (x / unit).ceil() * unit
}

/// A ceiling an [`Agreement`] missed.
#[derive(Debug, thiserror::Error, PartialEq)]
enum GateFailure {
    #[error("mean KL {got:.6} nats exceeds the ceiling {limit:.6}")]
    MeanKl { got: f64, limit: f64 },
    #[error(
        "the largest logit shift {got:.4} exceeds the recorded ceiling {limit:.4} (tie tolerance {:.4})",
        2.0 * limit
    )]
    Shift { got: f64, limit: f64 },
    #[error("the narrow logits differ from the exact ones by a non-finite amount")]
    NonFinite,
}

impl Gate {
    /// A gate with the mean-KL ceiling and the shift ceiling `max_shift`.
    const fn row(max_shift: f64) -> Self {
        Self {
            max_mean_kl: MAX_MEAN_KL,
            max_shift,
        }
    }

    /// The gate of a row whose narrow program measured the largest shift `measured`: the ceiling is
    /// `measured x SHIFT_MARGIN`, rounded up to two significant figures.
    fn measured(measured: f64) -> Self {
        Self::row(round_up_to_two_figures(measured * SHIFT_MARGIN))
    }

    fn check(&self, a: &Agreement) -> Result<(), GateFailure> {
        // A NaN or infinite measurement (a narrowing that overflowed) misses every ceiling.
        if !a.max_abs_logit.is_finite() || !a.mean_kl.is_finite() {
            return Err(GateFailure::NonFinite);
        }
        if a.mean_kl > self.max_mean_kl {
            return Err(GateFailure::MeanKl {
                got: a.mean_kl,
                limit: self.max_mean_kl,
            });
        }
        if a.max_abs_logit > self.max_shift {
            return Err(GateFailure::Shift {
                got: a.max_abs_logit,
                limit: self.max_shift,
            });
        }
        Ok(())
    }
}

/// `log_softmax` of `logits`, in f64.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
    let sum: f64 = logits.iter().map(|&x| f64::from(x - max).exp()).sum();
    let log_sum = sum.ln() + f64::from(max);
    logits.iter().map(|&x| f64::from(x) - log_sum).collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0
}

/// Compare the logits of `windows` windows (`[positions, n_vocab]` row-major each) the two programs produced.
fn compare(n_vocab: usize, windows: impl IntoIterator<Item = (Vec<f32>, Vec<f32>)>) -> Agreement {
    let (mut kl_sum, mut kl_max, mut agree, mut positions) = (0.0, 0.0f64, 0usize, 0usize);
    let mut max_abs_logit = 0.0f64;
    let (mut flips, mut max_flip_margin) = (0usize, 0.0f64);
    for (exact, narrow) in windows {
        assert_eq!(exact.len(), narrow.len());
        for (exact, narrow) in exact.chunks(n_vocab).zip(narrow.chunks(n_vocab)) {
            let (p, q) = (log_softmax(exact), log_softmax(narrow));
            let kl: f64 = p.iter().zip(&q).map(|(&p, &q)| p.exp() * (p - q)).sum();
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            let (top_exact, top_narrow) = (argmax(exact), argmax(narrow));
            if top_exact == top_narrow {
                agree += 1;
            } else {
                flips += 1;
                let margin = f64::from(exact[top_exact] - exact[top_narrow]);
                max_flip_margin = max_flip_margin.max(margin);
            }
            for (&e, &n) in exact.iter().zip(narrow) {
                // A NaN difference must survive `max`, which would drop it.
                let shift = f64::from(n - e).abs();
                max_abs_logit = if max_abs_logit.is_nan() || shift.is_nan() {
                    f64::NAN
                } else {
                    max_abs_logit.max(shift)
                };
            }
            positions += 1;
        }
    }
    let n = positions as f64;
    Agreement {
        positions,
        mean_kl: kl_sum / n,
        max_kl: kl_max,
        top1: agree as f64 / n,
        max_abs_logit,
        flips,
        max_flip_margin,
    }
}

/// What a compiled program declares about its operands: ops per [`OperandPrecision`].
fn precision_summary(program: &poot_graph_plan::staged::StagedProgram) -> BTreeMap<String, usize> {
    let mut summary = BTreeMap::new();
    for (_, _, stage) in program.stages() {
        for eqn in stage.graph().eqns() {
            if let Some(precision) = eqn.op.operand_precision() {
                *summary.entry(precision.to_string()).or_default() += 1;
            }
        }
    }
    summary
}

/// What one gate row measured.
#[derive(Debug)]
struct Measured {
    agreement: Agreement,
    exact_class: Tier2Class,
    narrow_class: Tier2Class,
    exact_ops: BTreeMap<String, usize>,
    narrow_ops: BTreeMap<String, usize>,
}

/// The agreement of the `Narrow` program with the `Exact` one of the checkpoint at `path` on `choice`, over the
/// first [`WINDOWS`] windows of [`CORPUS`] as the model's own tokenizer reads it; `None` when the device is
/// unavailable. `fragments` cuts the device's matrix units down to one dtype.
fn measure(
    path: &Path,
    choice: BackendChoice,
    backend: DeviceBackend,
    fragments: Option<FragmentDtype>,
) -> Option<Measured> {
    let exec = poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice))?;
    let handle = ModelHandle::load(path, &Registry::builtin().unwrap(), &Default::default())
        .expect("load the checkpoint");
    let tokens: Vec<i32> = handle
        .text()
        .encode(CORPUS)
        .expect("tokenize the corpus")
        .into_iter()
        .map(|t| t as i32)
        .collect();
    assert!(
        tokens.len() >= WINDOWS * N_CTX,
        "the corpus is {} tokens, the gate scores {}",
        tokens.len(),
        WINDOWS * N_CTX
    );
    let n_vocab = handle.config().vocab;
    let mut forced = TeacherForced::new(&handle, exec, N_CTX);
    let targets = forced.target_set();
    let options = CompileOptions::new(CompileLimits::STANDARD);
    let traced = forced.graph().clone();
    let (exact_entry, exact) = forced.add_variant(&traced, &targets, &options);
    let narrow_options = CompileOptions {
        fragment_only: fragments,
        ..options
            .clone()
            .with_matmul_precision(MatmulPrecision::Narrow)
    };
    let (narrow_entry, narrow) = forced.add_variant(&traced, &targets, &narrow_options);
    let windows = (0..WINDOWS).map(|w| {
        let window = &tokens[w * N_CTX..][..N_CTX];
        (
            forced.logits_of(exact_entry, window),
            forced.logits_of(narrow_entry, window),
        )
    });
    let agreement = compare(n_vocab, windows.collect::<Vec<_>>());
    Some(Measured {
        agreement,
        exact_class: exact.numerics_class(),
        narrow_class: narrow.numerics_class(),
        exact_ops: precision_summary(&exact),
        narrow_ops: precision_summary(&narrow),
    })
}

/// The narrowed dtypes a program's ops declare, by their printed name.
fn narrowed(ops: &BTreeMap<String, usize>, dtype: FragmentDtype) -> usize {
    ops.get(&OperandPrecision::Narrow(dtype).to_string())
        .copied()
        .unwrap_or(0)
}

/// One checkpoint row: the narrow program must really narrow `contraction` operands, the exact one must not,
/// and the agreement must meet `gate`.
fn row(
    path: &Path,
    choice: BackendChoice,
    backend: DeviceBackend,
    fragments: Option<FragmentDtype>,
    contraction: FragmentDtype,
    gate: Gate,
) {
    let Some(measured) = measure(path, choice, backend, fragments) else {
        return;
    };
    let a = &measured.agreement;
    eprintln!(
        "NARROW-GATE {} {choice} contraction={contraction:?} positions={} top1={:.5} mean_kl={:.3e} max_kl={:.3e} \
         max_abs_logit={:.4} shift_ceiling={} flips={} max_flip_margin={:.4} exact_class={:?} narrow_class={:?} \
         exact_ops={:?} narrow_ops={:?}",
        path.display(),
        a.positions,
        a.top1,
        a.mean_kl,
        a.max_kl,
        a.max_abs_logit,
        gate.max_shift,
        a.flips,
        a.max_flip_margin,
        measured.exact_class,
        measured.narrow_class,
        measured.exact_ops,
        measured.narrow_ops,
    );
    assert_eq!(a.positions, WINDOWS * N_CTX);
    assert!(
        measured.exact_ops.keys().all(|k| k == "exact"),
        "the exact program narrowed: {:?}",
        measured.exact_ops
    );
    assert!(
        narrowed(&measured.narrow_ops, contraction) > 0,
        "the narrow program narrows no op to {contraction:?}: {:?}",
        measured.narrow_ops
    );
    if let Some(only) = fragments {
        let other = match only {
            FragmentDtype::Bf16 => FragmentDtype::F16,
            FragmentDtype::F16 => FragmentDtype::Bf16,
        };
        assert_eq!(
            narrowed(&measured.narrow_ops, other),
            0,
            "the row restricts the planner to {only:?} fragments, yet an op narrows to {other:?}: {:?}",
            measured.narrow_ops
        );
    }
    if let Err(failure) = gate.check(a) {
        panic!("{}: {failure} ({a:?})", path.display());
    }
}

/// One module of rows per (backend, contraction dtype): every checkpoint on that device, each with the largest
/// logit shift its narrow program measured on Strix Halo gfx1151 (POOT-1454; deterministic on the device), from
/// which [`Gate::measured`] records the row's ceiling.
macro_rules! gate_rows {
    (
        $module:ident, $choice:expr, $backend:expr, $fragments:expr, $contraction:expr;
        $($(#[$row_attr:meta])* $name:ident => $checkpoint:literal, $measured_shift:literal,)*
    ) => {
        mod $module {
            use super::*;

            $(
                #[test]
                $(#[$row_attr])*
                fn $name() {
                    let Some(path) =
                        poot_test_util::model_path(poot_test_util::checkpoint!($checkpoint))
                    else {
                        return;
                    };
                    row(
                        &path,
                        $choice,
                        $backend,
                        $fragments,
                        $contraction,
                        Gate::measured($measured_shift),
                    );
                }
            )*
        }
    };
}

gate_rows! {
    // The contraction dtype on ROCm: bf16, range-safe (POOT-1454 decision).
    rocm_bf16, BackendChoice::Rocm, DeviceBackend::Rocm, None, FragmentDtype::Bf16;
    qwen3_0_6b => "qwen3-0.6b", 0.2046,
    qwen3_4b => "qwen3-4b", 0.1503,
    qwen3_0_6b_q8_0 => "qwen3-0.6b-gguf/Qwen3-0.6B-Q8_0.gguf", 0.3240,
    qwen2_5_1_5b_q4_k_m => "qwen2.5-1.5b-q4km/qwen2.5-1.5b-instruct-q4_k_m.gguf", 0.4863,
    gemma_3_1b_it => "gemma-3-1b-it", 2.3151,
    llama_3_2_1b_instruct => "llama-3.2-1b-instruct", 0.1328,
}

gate_rows! {
    // The f16 contraction variant on ROCm, kept as the record of the dtype comparison (1429 Q3). Its activations are
    // row-scaled (POOT-1457); in range the scale is exactly 1, so the shifts are the unguarded f16 program's.
    rocm_f16, BackendChoice::Rocm, DeviceBackend::Rocm, Some(FragmentDtype::F16), FragmentDtype::F16;
    qwen3_0_6b => "qwen3-0.6b", 0.0331,
    qwen3_4b => "qwen3-4b", 0.0326,
    qwen3_0_6b_q8_0 => "qwen3-0.6b-gguf/Qwen3-0.6B-Q8_0.gguf", 0.0693,
    qwen2_5_1_5b_q4_k_m => "qwen2.5-1.5b-q4km/qwen2.5-1.5b-instruct-q4_k_m.gguf", 0.0690,
    gemma_3_1b_it => "gemma-3-1b-it", 0.6164,
    llama_3_2_1b_instruct => "llama-3.2-1b-instruct", 0.0257,
}

gate_rows! {
    // NVPTX (sm_80+) has bf16 fragments: bf16 contractions, as on ROCm. Measured on a RunPod RTX 4090 pod
    // (POOT-1454, scripts/ptx-pod-lane --models scripts/ptx-pod-models/narrow-gate.tsv). Gemma 3 misses the mean-KL
    // ceiling here (1.86e-3) and is parked (checkpoint-ptx.parked.txt): receipts/1454-narrow-dtype-policy/design.md
    // section 10.
    ptx_bf16, BackendChoice::Ptx, DeviceBackend::Ptx, None, FragmentDtype::Bf16;
    qwen3_0_6b => "qwen3-0.6b", 0.4419,
    qwen3_4b => "qwen3-4b", 0.1410,
    qwen3_0_6b_q8_0 => "qwen3-0.6b-gguf/Qwen3-0.6B-Q8_0.gguf", 0.4624,
    qwen2_5_1_5b_q4_k_m => "qwen2.5-1.5b-q4km/qwen2.5-1.5b-instruct-q4_k_m.gguf", 0.4799,
    #[ignore = "parked for POOT-1457 (f16 contractions with the row-scale guard): under bf16 on PTX the mean KL is \
                1.86e-3 > 1e-3, bf16 rounding-boundary amplification (PTX/ROCm worst-logit ratio 3.4)"]
    gemma_3_1b_it => "gemma-3-1b-it", 7.9349,
    llama_3_2_1b_instruct => "llama-3.2-1b-instruct", 0.1344,
}

gate_rows! {
    wgpu_f16, BackendChoice::Wgpu, DeviceBackend::Wgpu, None, FragmentDtype::F16;
    qwen3_0_6b => "qwen3-0.6b", 0.0255,
    qwen3_4b => "qwen3-4b", 0.0345,
    qwen3_0_6b_q8_0 => "qwen3-0.6b-gguf/Qwen3-0.6B-Q8_0.gguf", 0.0667,
    qwen2_5_1_5b_q4_k_m => "qwen2.5-1.5b-q4km/qwen2.5-1.5b-instruct-q4_k_m.gguf", 0.0668,
    gemma_3_1b_it => "gemma-3-1b-it", 0.5030,
    llama_3_2_1b_instruct => "llama-3.2-1b-instruct", 0.0242,
}

gate_rows! {
    vulkan_f16, BackendChoice::Vulkan, DeviceBackend::Vulkan, None, FragmentDtype::F16;
    qwen3_0_6b => "qwen3-0.6b", 0.0599,
    qwen3_4b => "qwen3-4b", 0.0245,
    qwen3_0_6b_q8_0 => "qwen3-0.6b-gguf/Qwen3-0.6B-Q8_0.gguf", 0.0653,
    qwen2_5_1_5b_q4_k_m => "qwen2.5-1.5b-q4km/qwen2.5-1.5b-instruct-q4_k_m.gguf", 0.0692,
    gemma_3_1b_it => "gemma-3-1b-it", 0.4350,
    llama_3_2_1b_instruct => "llama-3.2-1b-instruct", 0.0176,
}

// ---------------------------------------------------------------------------------------------
// Model-free: the metrics and the gate, on distributions built in the test.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod metric_tests {
    use super::*;

    const VOCAB: usize = 16;

    /// A window of `positions` rows whose logit `(p + v) % VOCAB` is the largest.
    fn rows(positions: usize) -> Vec<f32> {
        (0..positions)
            .flat_map(|p| (0..VOCAB).map(move |v| ((p * 3 + v) % VOCAB) as f32 * 0.5))
            .collect()
    }

    /// The exact window [`agreement`] compares against: 200 positions.
    const POSITIONS: usize = 200;

    fn agreement(narrow: &[f32]) -> Agreement {
        compare(VOCAB, [(rows(POSITIONS), narrow.to_vec())])
    }

    /// `rows(POSITIONS)` with position 0's top token lowered and its runner-up raised by `by`: the two swap rank
    /// once `2 * by` passes the exact top-two gap, 0.5.
    fn flipped_at_position_0(by: f32) -> Vec<f32> {
        let mut narrow = rows(POSITIONS);
        let top = argmax(&narrow[..VOCAB]);
        let runner_up = (top + VOCAB - 1) % VOCAB;
        narrow[top] -= by;
        narrow[runner_up] += by;
        narrow
    }

    #[test]
    fn the_exact_logits_agree_with_themselves_and_pass() {
        let a = agreement(&rows(POSITIONS));
        assert_eq!((a.top1, a.mean_kl, a.max_abs_logit), (1.0, 0.0, 0.0));
        assert_eq!((a.flips, a.max_flip_margin), (0, 0.0));
        assert_eq!(Gate::row(0.1).check(&a), Ok(()));
    }

    /// The tie rule (ADR-0101 decision 3): a flip where the exact top-two gap (0.5) is within the tie tolerance
    /// (2 x the ceiling 0.31 = 0.62) is allowed, whatever the top-1 fraction (here 0.995 exactly, and a single flip
    /// in 20 positions would be 0.95). Mutation: put a top-1 floor of 0.999 back into `Gate::check`; this row
    /// goes red.
    #[test]
    fn a_flip_within_the_tie_tolerance_passes() {
        let a = agreement(&flipped_at_position_0(0.3));
        assert_eq!(a.flips, 1, "{a:?}");
        assert_eq!(a.max_flip_margin, 0.5);
        assert!((a.max_abs_logit - 0.3).abs() < 1e-6, "{a:?}");
        assert!(a.top1 < 1.0);
        // 0.31, not 0.3: the shift is 0.3 rounded in f32 (0.30000019).
        assert_eq!(Gate::row(0.31).check(&a), Ok(()));
    }

    /// A flip where the exact top-two gap (0.5) is beyond the tie tolerance (2 x the ceiling 0.2 = 0.4) is
    /// refused: it needs a shift past the ceiling (module docs), and the shift ceiling refuses it. Mutation: drop
    /// the `max_shift` comparison in `Gate::check`; this row goes red.
    #[test]
    fn a_flip_beyond_the_tie_tolerance_is_refused_by_the_shift_ceiling() {
        let gate = Gate::row(0.2);
        let a = agreement(&flipped_at_position_0(0.3));
        assert_eq!(a.flips, 1, "{a:?}");
        assert!(a.max_flip_margin > 2.0 * gate.max_shift, "{a:?}");
        assert!(
            matches!(gate.check(&a), Err(GateFailure::Shift { .. })),
            "{:?}",
            gate.check(&a)
        );
    }

    /// Mutation: drop the `mean_kl` comparison in `Gate::check`.
    #[test]
    fn a_shifted_distribution_misses_the_kl_ceiling_with_the_ranking_intact() {
        // Scale every logit: the arg-max stays, the distribution sharpens.
        let narrow: Vec<f32> = rows(POSITIONS).iter().map(|x| x * 1.2).collect();
        let a = agreement(&narrow);
        assert_eq!(a.top1, 1.0);
        assert!(a.mean_kl > MAX_MEAN_KL, "{a:?}");
        // A ceiling the shift (1.5) meets, so the KL is what fails.
        assert!(matches!(
            Gate::row(10.0).check(&a),
            Err(GateFailure::MeanKl { .. })
        ));
    }

    /// A narrowing that overflowed produces infinity or NaN: it fails the gate, it is not skipped by `max`.
    #[test]
    fn a_non_finite_logit_fails_the_gate() {
        for bad in [f32::NAN, f32::INFINITY] {
            let mut narrow = rows(POSITIONS);
            narrow[5] = bad;
            let a = agreement(&narrow);
            assert_eq!(
                Gate::row(10.0).check(&a),
                Err(GateFailure::NonFinite),
                "{bad}: {a:?}"
            );
        }
    }

    /// The recorded ceiling is the measured shift times the margin, rounded up to two figures, never down.
    /// Mutation: round to nearest (`round` for `ceil`) in `round_up_to_two_figures`; 0.2046 x 2.4 = 0.491 gives
    /// 0.49, below the derivation's 0.50, and this row goes red.
    #[test]
    fn a_ceiling_is_the_measured_shift_times_the_margin_rounded_up() {
        for (measured, ceiling) in [
            (0.2046, 0.50),
            (2.3151, 5.6),
            (0.0176, 0.043),
            (0.0326, 0.079),
        ] {
            let got = Gate::measured(measured).max_shift;
            assert!(
                (got - ceiling).abs() < 1e-9,
                "{measured}: {got} != {ceiling}"
            );
            assert!(got >= measured * SHIFT_MARGIN, "{measured}: {got}");
        }
    }

    #[test]
    fn the_largest_logit_shift_is_recorded() {
        let mut narrow = rows(POSITIONS);
        narrow[7] += 0.25;
        let a = agreement(&narrow);
        assert_eq!(a.max_abs_logit, 0.25);
        assert_eq!(a.flips, 0);
    }
}
