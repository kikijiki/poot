//! POOT-1542 SC-5: the real `TQ1_0` checkpoint against llama.cpp.
//!
//! `scripts/make-tq1-checkpoint.sh` fetches the file (sha256-checked) and makes the f16 copies. The model is TriLM
//! 1.1B (`TriLM_1.1B_Unpacked-tq1_0.gguf`, Mungert/TriLM_1.1B_Unpacked-GGUF), a model trained with ternary weights,
//! so `TQ1_0` holds its weights as they are rather than as a lossy re-quantization. It has 168 `TQ1_0` tensors (the
//! seven projections of each of its 24 layers); the embedding is `Q4_K`, the output `Q6_K`, the norms F32. The
//! `TQ1_0` dequantization is pinned elsewhere, bit for bit, to llama.cpp's `to_float`
//! (`flash_next_dequant::dq6_tq1_0_real_blocks_decode_to_the_reference`, and the card's receipt over every tensor).
//! This module scores the whole model: the teacher-forced gate of `llamacpp_kl_tests` (the same corpus, window and
//! metrics) on the CPU oracle and on the ROCm and wgpu executors, against `llama-perplexity` on the same GGUF.
//!
//! The reference binary is the box's `llama-perplexity` at commit `cd959ba`, as the `IQ1` rows use.
//!
//! llama.cpp multiplies the `TQ1_0` weights with activations it quantizes to 8 bits (`Q8_K`); poot multiplies the
//! decoded f32 weights with f32 activations. So scoring against llama.cpp's own `TQ1_0` run measures the engines'
//! arithmetic as well as the decode. The second set of rows removes it: the reference is llama.cpp running an f16
//! copy of the same file (`llama-quantize --allow-requantize <gguf> <copy> F16`, every tensor widened by llama.cpp's
//! own `to_float`), whose matmuls take f16 weights and unquantized activations. Measured, `cd959ba`, windows of 128
//! tokens, 2026-11-16 (`Agreement`: mean KL, max KL, NLL delta, top-1, top-5 overlap; [`Engine`] in the first
//! column):
//!
//! | reference                        | engine (windows) | mean KL | max KL  | NLL delta | top-1 | top-5 |
//! |----------------------------------|------------------|---------|---------|-----------|-------|-------|
//! | llama.cpp TQ1_0                  | oracle (1)       | 0.00326 | 0.00839 | -0.01657  | 0.984 | 0.978 |
//! | llama.cpp TQ1_0                  | ROCm (5)         | 0.00314 | 0.01327 | -0.00608  | 0.975 | 0.966 |
//! | llama.cpp TQ1_0                  | wgpu (5)         | 0.00314 | 0.01323 | -0.00605  | 0.975 | 0.966 |
//! | llama.cpp f16 copy               | oracle (1)       | 3.9e-7  | 1.3e-6  | +0.00018  | 1.000 | 1.000 |
//! | llama.cpp f16 copy               | ROCm (5)         | 2.2e-7  | 1.8e-6  | +0.00008  | 1.000 | 1.000 |
//! | llama.cpp f16 copy               | wgpu (5)         | 3.8e-7  | 2.0e-6  | +0.00011  | 1.000 | 1.000 |
//! | control: Q4_0 f16 copy           | oracle (1)       | 0.02796 | 0.12349 | -0.03118  | 0.873 | 0.895 |
//!
//! The 0.003 mean KL against llama.cpp's own `TQ1_0` run is its activation quantization, not the decode: the weights
//! are bit-equal to llama.cpp's, and against the f16 copy of those same weights poot is within 4e-7 mean KL on
//! every engine. Each gate holds the worst measurement with room: 30% on the mean KL of [`GATE`], a factor of
//! four on [`F16_GATE`], which the control exceeds by four orders of magnitude.
//!
//! Mutation: move the second segment of `TQ1_0`'s codes in `blocks/lattice.rs` from element 160 to element 128; the
//! f16 rows go red.

use std::path::Path;

use poot_models::registry::Registry;
use poot_runtime_common::DeviceBackend;

use super::llamacpp_kl_tests::{
    Agreement, CORPUS, Gate, Reference, TeacherForced, compare_chunks, run_reference_at,
};
use super::llamacpp_longrope_tests::OracleForced;
use super::{BackendChoice, ModelHandle, open_executor};

/// The checkpoint's file stem under `$POOT_MODELS_DIR/tq1/`.
const MODEL: &str = "TriLM_1.1B_Unpacked-tq1_0";

/// The `llama-perplexity` these rows run against.
const LLAMA_CPP_COMMIT: &str = "cd959ba";

const N_CTX: usize = 128;

/// The windows the CPU oracle scores: one, since an oracle window is minutes.
const ORACLE_WINDOWS: usize = 1;

/// The windows a device scores: the corpus holds five.
const DEVICE_WINDOWS: usize = 5;

/// The gate on `TQ1_0` against llama.cpp, from the measurements in the module comment.
const GATE: Gate = Gate {
    max_mean_kl: 0.0043,
    max_max_kl: 0.0175,
    max_abs_nll_delta: 0.022,
    min_top1: 0.93,
};

/// poot's logits for a window, `[N_CTX, n_vocab]` row-major.
type WindowLogits = Box<dyn FnMut(&[i32]) -> Vec<f32>>;

#[derive(Clone, Copy, Debug)]
enum Engine {
    Oracle,
    Device(BackendChoice, DeviceBackend),
}

fn checkpoint() -> Option<std::path::PathBuf> {
    let dir = std::env::var("POOT_MODELS_DIR")
        .ok()
        .filter(|d| !d.is_empty());
    let Some(dir) = dir else {
        poot_test_util::missing_prerequisite(&"POOT_MODELS_DIR is unset");
        return None;
    };
    let path = Path::new(&dir).join(format!("tq1/{MODEL}.gguf"));
    if !path.exists() {
        poot_test_util::missing_prerequisite(&format_args!("{} is absent", path.display()));
        return None;
    }
    Some(path)
}

fn load(gguf: &Path) -> ModelHandle {
    ModelHandle::load(gguf, &Registry::builtin().unwrap(), &Default::default())
        .expect("load the GGUF")
}

/// poot's agreement, running `scored` on `engine`, with llama.cpp's reference over [`CORPUS`] computed on
/// `reference`. `None` when the device or `llama-perplexity` is unavailable.
fn measure(reference: &Path, scored: &Path, engine: Engine, windows: usize) -> Option<Agreement> {
    let exec = match engine {
        Engine::Oracle => None,
        Engine::Device(choice, backend) => Some(poot_runtime_common::device_skip::open_or_skip(
            backend,
            open_executor(choice),
        )?),
    };
    let scratch = poot_test_util::unique_temp_path("reference.bin");
    let corpus = scratch.with_file_name("corpus.txt");
    std::fs::write(&corpus, CORPUS).unwrap();
    run_reference_at(LLAMA_CPP_COMMIT, reference, &corpus, N_CTX, &scratch)?;
    let bytes = std::fs::read(&scratch).unwrap();
    let reference = Reference::parse(&bytes).expect("parse the reference");
    assert_eq!(reference.n_ctx, N_CTX);
    let handle = load(scored);
    assert_eq!(handle.config().vocab, reference.n_vocab);
    let first = reference.window_tokens(0).to_vec();
    let mut logits: WindowLogits = match exec {
        None => {
            let mut oracle = OracleForced::new(&handle, &first);
            Box::new(move |window| oracle.logits(window))
        }
        Some(exec) => {
            let mut forced = TeacherForced::new(&handle, exec, N_CTX);
            Box::new(move |window| forced.logits(window))
        }
    };
    let agreement = compare_chunks(&reference, windows, |window| logits(window));
    assert_eq!(agreement.positions, windows * (N_CTX - 1 - N_CTX / 2));
    Some(agreement)
}

fn check(engine: Engine, windows: usize) {
    let Some(gguf) = checkpoint() else {
        return;
    };
    let Some(agreement) = measure(&gguf, &gguf, engine, windows) else {
        return;
    };
    eprintln!(
        "TQ1_0 on {engine:?}, {windows} windows, llama.cpp {LLAMA_CPP_COMMIT}: {agreement:?}"
    );
    if let Err(failure) = GATE.check(&agreement) {
        panic!("TQ1_0 on {engine:?}: {failure} ({agreement:?})");
    }
}

#[test]
fn tq1_0_follows_llamacpp_on_the_cpu_oracle() {
    check(Engine::Oracle, ORACLE_WINDOWS);
}

#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; loads a 1.1B TQ1_0 GGUF; the checkpoint-rocm lane runs it by name"]
fn tq1_0_follows_llamacpp_on_rocm() {
    check(
        Engine::Device(BackendChoice::Rocm, DeviceBackend::Rocm),
        DEVICE_WINDOWS,
    );
}

#[test]
#[ignore = "needs the wgpu device and /tmp/poot-gpu.lock; loads a 1.1B TQ1_0 GGUF; the checkpoint-wgpu lane runs it by name"]
fn tq1_0_follows_llamacpp_on_wgpu() {
    check(
        Engine::Device(BackendChoice::Wgpu, DeviceBackend::Wgpu),
        DEVICE_WINDOWS,
    );
}

/// The path of the f16 copy of the GGUF `stem`: `llama-quantize --allow-requantize <gguf> <copy> F16`, which widens
/// every quantized tensor with llama.cpp's own `to_float` and rounds it to f16, so the copy holds llama.cpp's
/// dequantization of the `TQ1_0` model (and of its `Q4_K` embedding and `Q6_K` output).
fn f16_copy(stem: &str) -> Option<std::path::PathBuf> {
    let dir = std::env::var("POOT_MODELS_DIR")
        .ok()
        .filter(|d| !d.is_empty())?;
    let path = Path::new(&dir).join(format!("tq1/f16/{stem}-as-F16.gguf"));
    if !path.exists() {
        poot_test_util::missing_prerequisite(&format_args!("{} is absent", path.display()));
        return None;
    }
    Some(path)
}

/// The gate on poot's `TQ1_0` model against llama.cpp running the f16 copy of the same dequantized weights.
const F16_GATE: Gate = Gate {
    max_mean_kl: 1.5e-6,
    max_max_kl: 8e-6,
    max_abs_nll_delta: 4e-4,
    min_top1: 0.99,
};

fn check_against_f16_copy(engine: Engine, windows: usize) {
    let (Some(gguf), Some(copy)) = (checkpoint(), f16_copy(MODEL)) else {
        return;
    };
    let Some(agreement) = measure(&copy, &gguf, engine, windows) else {
        return;
    };
    eprintln!(
        "TQ1_0 against llama.cpp's f16 copy, on {engine:?}, {windows} windows: {agreement:?}"
    );
    if let Err(failure) = F16_GATE.check(&agreement) {
        panic!("TQ1_0 on {engine:?} against the f16 copy: {failure} ({agreement:?})");
    }
}

#[test]
fn tq1_0_matches_llamacpp_running_its_f16_dequantization_on_the_cpu_oracle() {
    check_against_f16_copy(Engine::Oracle, ORACLE_WINDOWS);
}

#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; loads a 1.1B TQ1_0 GGUF; the checkpoint-rocm lane runs it by name"]
fn tq1_0_matches_llamacpp_running_its_f16_dequantization_on_rocm() {
    check_against_f16_copy(
        Engine::Device(BackendChoice::Rocm, DeviceBackend::Rocm),
        DEVICE_WINDOWS,
    );
}

#[test]
#[ignore = "needs the wgpu device and /tmp/poot-gpu.lock; loads a 1.1B TQ1_0 GGUF; the checkpoint-wgpu lane runs it by name"]
fn tq1_0_matches_llamacpp_running_its_f16_dequantization_on_wgpu() {
    check_against_f16_copy(
        Engine::Device(BackendChoice::Wgpu, DeviceBackend::Wgpu),
        DEVICE_WINDOWS,
    );
}

/// The control: poot scores the `TQ1_0` file against llama.cpp's reference of the `Q4_0` file of the same model. A
/// different quantization of the same weights is a decode that is wrong in its values, and the f16 gate sees it.
#[test]
fn a_different_quantization_leaves_the_f16_gate_on_the_cpu_oracle() {
    let (Some(tq1), Some(q4_0_copy)) = (checkpoint(), f16_copy("TriLM_1.1B_Unpacked-q4_0")) else {
        return;
    };
    let Some(agreement) = measure(&q4_0_copy, &tq1, Engine::Oracle, ORACLE_WINDOWS) else {
        return;
    };
    eprintln!("TQ1_0 scored against the Q4_0 f16 copy: {agreement:?}");
    assert!(
        F16_GATE.check(&agreement).is_err(),
        "the f16 gate passed a different quantization: {agreement:?}"
    );
}
