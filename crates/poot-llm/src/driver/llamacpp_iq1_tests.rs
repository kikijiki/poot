//! POOT-1541 SC-4: the real `IQ1_S` and `IQ1_M` checkpoints against llama.cpp.
//!
//! `scripts/make-iq1-checkpoints.sh` fetches the files (sha256-checked) and makes the f16 copies. The two GGUFs are `Qwen2.5-0.5B-Instruct-IQ1_S.gguf` and `-IQ1_M.gguf` (ThomasBaruzier/Qwen2.5-0.5B-Instruct-GGUF),
//! each holding 21 IQ1 tensors (`blk.2..=23.ffn_down` and neighbours), the rest `IQ4_NL`, `Q5_0`, `Q2_K` and
//! `Q8_0`. Their dequantization is pinned elsewhere, bit for bit, to llama.cpp's `to_float`
//! (`flash_next_dequant::dq5_iq1_real_blocks_decode_to_the_reference`, and the card's receipt over every tensor). This
//! module scores the whole model: the teacher-forced gate of `llamacpp_kl_tests` (the same corpus, window and
//! metrics) on the CPU oracle and on the ROCm and wgpu executors, against `llama-perplexity` on the same GGUF.
//!
//! The reference binary is the box's `llama-perplexity` at commit `cd959ba`, not the commit the Q8_0 rows pin:
//! those gates were measured against `d812350` and say so. These were measured against `cd959ba`.
//!
//! llama.cpp multiplies the IQ1 weights with activations it quantizes to 8 bits (`Q8_K`); poot multiplies the
//! decoded f32 weights with f32 activations. So scoring against llama.cpp's own IQ1 run measures the engines'
//! arithmetic as well as the decode, and a model of 1.5 bits per weight amplifies the difference. The second set of
//! rows removes it: the reference is llama.cpp running an f16 copy of the same file (`llama-quantize
//! --allow-requantize <gguf> <copy> F16`, every tensor widened by llama.cpp's own `to_float`), whose matmuls take
//! f16 weights and unquantized activations. Measured, `cd959ba`, windows of 128 tokens, 2026-11-15
//! (`Agreement`: mean KL, max KL, NLL delta, top-1, top-5 overlap; [`Engine`] in the first column):
//!
//! | model, reference                  | engine (windows)  | mean KL | max KL  | NLL delta | top-1 | top-5 |
//! |-----------------------------------|-------------------|---------|---------|-----------|-------|-------|
//! | IQ1_S, llama.cpp IQ1_S            | oracle (1)        | 0.00469 | 0.00911 | +0.00258  | 0.984 | 0.949 |
//! | IQ1_S, llama.cpp IQ1_S            | ROCm (5)          | 0.00519 | 0.04131 | -0.00165  | 0.946 | 0.955 |
//! | IQ1_S, llama.cpp IQ1_S            | wgpu (5)          | 0.00531 | 0.04043 | -0.00186  | 0.949 | 0.954 |
//! | IQ1_M, llama.cpp IQ1_M            | oracle (1)        | 0.00447 | 0.02331 | -0.00089  | 1.000 | 0.959 |
//! | IQ1_M, llama.cpp IQ1_M            | ROCm (5)          | 0.00389 | 0.02210 | +0.00316  | 0.959 | 0.958 |
//! | IQ1_M, llama.cpp IQ1_M            | wgpu (5)          | 0.00390 | 0.02235 | +0.00292  | 0.959 | 0.959 |
//! | IQ1_S, llama.cpp f16 copy         | oracle (1)        | 1.0e-5  | 1.3e-4  | -0.00105  | 1.000 | 1.000 |
//! | IQ1_S, llama.cpp f16 copy         | ROCm (5)          | 4.2e-6  | 8.8e-5  | -0.00013  | 0.997 | 0.999 |
//! | IQ1_M, llama.cpp f16 copy         | oracle (1)        | 5.6e-6  | 3.0e-5  | +0.00009  | 1.000 | 1.000 |
//! | IQ1_M, llama.cpp f16 copy         | ROCm (5)          | 3.1e-6  | 3.4e-5  | +0.00013  | 0.997 | 0.999 |
//! | control: IQ1_S, IQ1_M f16 copy    | oracle (1)        | 0.410   | 1.14    | -0.0950   | 0.667 | 0.625 |
//!
//! The 0.004 to 0.005 mean KL against llama.cpp's own IQ1 run is its activation quantization, not the decode: the
//! weights are bit-equal to llama.cpp's (every IQ1 tensor of both files, receipt `1541-iq1`), and against the f16
//! copy of those same weights poot is within 1e-5 mean KL on every engine. The same ordinary-quantization noise
//! is what the `Q8_0` and `Q4_K_M` rows of `llamacpp_kl_tests` measure (0.0015 to 0.0023). Each gate holds the
//! worst measurement with room: 30% on the mean KL of [`GATE`], a factor of three on [`F16_GATE`], which the
//! control exceeds by four orders of magnitude.
//!
//! Mutation: change the shift's exponent in `blocks/lattice.rs` from -3 to -2; the f16 rows go red.

use std::path::Path;

use poot_models::registry::Registry;
use poot_runtime_common::DeviceBackend;

use super::llamacpp_kl_tests::{
    Agreement, CORPUS, Gate, Reference, TeacherForced, compare_chunks, run_reference_at,
};
use super::llamacpp_longrope_tests::OracleForced;
use super::{BackendChoice, ModelHandle, open_executor};

/// The `llama-perplexity` these rows run against.
const LLAMA_CPP_COMMIT: &str = "cd959ba";

const N_CTX: usize = 128;

/// The windows the CPU oracle scores: one, since an oracle window is minutes.
const ORACLE_WINDOWS: usize = 1;

/// The windows a device scores: the corpus holds five.
const DEVICE_WINDOWS: usize = 5;

/// The gate on `IQ1_S` and `IQ1_M` against llama.cpp, from the measurements in the module comment.
const GATE: Gate = Gate {
    max_mean_kl: 0.007,
    max_max_kl: 0.055,
    max_abs_nll_delta: 0.0045,
    min_top1: 0.92,
};

/// poot's logits for a window, `[N_CTX, n_vocab]` row-major.
type WindowLogits = Box<dyn FnMut(&[i32]) -> Vec<f32>>;

#[derive(Clone, Copy, Debug)]
enum Engine {
    Oracle,
    Device(BackendChoice, DeviceBackend),
}

fn checkpoint(quant: &str) -> Option<std::path::PathBuf> {
    let dir = std::env::var("POOT_MODELS_DIR")
        .ok()
        .filter(|d| !d.is_empty());
    let Some(dir) = dir else {
        poot_test_util::missing_prerequisite(&"POOT_MODELS_DIR is unset");
        return None;
    };
    let path = Path::new(&dir).join(format!("iq1/Qwen2.5-0.5B-Instruct-{quant}.gguf"));
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

fn check(quant: &str, engine: Engine, windows: usize) {
    let Some(gguf) = checkpoint(quant) else {
        return;
    };
    let Some(agreement) = measure(&gguf, &gguf, engine, windows) else {
        return;
    };
    eprintln!(
        "{quant} on {engine:?}, {windows} windows, llama.cpp {LLAMA_CPP_COMMIT}: {agreement:?}"
    );
    if let Err(failure) = GATE.check(&agreement) {
        panic!("{quant} on {engine:?}: {failure} ({agreement:?})");
    }
}

#[test]
fn iq1_s_follows_llamacpp_on_the_cpu_oracle() {
    check("IQ1_S", Engine::Oracle, ORACLE_WINDOWS);
}

#[test]
fn iq1_m_follows_llamacpp_on_the_cpu_oracle() {
    check("IQ1_M", Engine::Oracle, ORACLE_WINDOWS);
}

#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; loads a 0.5B IQ1 GGUF; the checkpoint-rocm lane runs it by name"]
fn iq1_s_follows_llamacpp_on_rocm() {
    check(
        "IQ1_S",
        Engine::Device(BackendChoice::Rocm, DeviceBackend::Rocm),
        DEVICE_WINDOWS,
    );
}

#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; loads a 0.5B IQ1 GGUF; the checkpoint-rocm lane runs it by name"]
fn iq1_m_follows_llamacpp_on_rocm() {
    check(
        "IQ1_M",
        Engine::Device(BackendChoice::Rocm, DeviceBackend::Rocm),
        DEVICE_WINDOWS,
    );
}

#[test]
#[ignore = "needs the wgpu device and /tmp/poot-gpu.lock; loads a 0.5B IQ1 GGUF; the checkpoint-wgpu lane runs it by name"]
fn iq1_s_follows_llamacpp_on_wgpu() {
    check(
        "IQ1_S",
        Engine::Device(BackendChoice::Wgpu, DeviceBackend::Wgpu),
        DEVICE_WINDOWS,
    );
}

#[test]
#[ignore = "needs the wgpu device and /tmp/poot-gpu.lock; loads a 0.5B IQ1 GGUF; the checkpoint-wgpu lane runs it by name"]
fn iq1_m_follows_llamacpp_on_wgpu() {
    check(
        "IQ1_M",
        Engine::Device(BackendChoice::Wgpu, DeviceBackend::Wgpu),
        DEVICE_WINDOWS,
    );
}

/// The path of the f16 copy of `quant`'s GGUF: `llama-quantize --allow-requantize <gguf> <copy> F16`, which widens
/// every quantized tensor with llama.cpp's own `to_float` and rounds it to f16, so the copy holds llama.cpp's
/// dequantization of the IQ1 model (and of its `IQ4_NL`, `Q5_0`, `Q2_K` and `Q8_0` tensors).
fn f16_copy(quant: &str) -> Option<std::path::PathBuf> {
    let dir = std::env::var("POOT_MODELS_DIR")
        .ok()
        .filter(|d| !d.is_empty())?;
    let path = Path::new(&dir).join(format!("iq1/f16/Qwen2.5-0.5B-Instruct-{quant}-as-F16.gguf"));
    if !path.exists() {
        poot_test_util::missing_prerequisite(&format_args!("{} is absent", path.display()));
        return None;
    }
    Some(path)
}

/// The gate on poot's IQ1 model against llama.cpp running the f16 copy of the same dequantized weights.
const F16_GATE: Gate = Gate {
    max_mean_kl: 3e-5,
    max_max_kl: 5e-4,
    max_abs_nll_delta: 3e-3,
    min_top1: 0.99,
};

fn check_against_f16_copy(quant: &str, engine: Engine, windows: usize) {
    let (Some(gguf), Some(copy)) = (checkpoint(quant), f16_copy(quant)) else {
        return;
    };
    let Some(agreement) = measure(&copy, &gguf, engine, windows) else {
        return;
    };
    eprintln!(
        "{quant} against llama.cpp's f16 copy, on {engine:?}, {windows} windows: {agreement:?}"
    );
    if let Err(failure) = F16_GATE.check(&agreement) {
        panic!("{quant} on {engine:?} against the f16 copy: {failure} ({agreement:?})");
    }
}

#[test]
fn iq1_s_matches_llamacpp_running_its_f16_dequantization_on_the_cpu_oracle() {
    check_against_f16_copy("IQ1_S", Engine::Oracle, ORACLE_WINDOWS);
}

#[test]
fn iq1_m_matches_llamacpp_running_its_f16_dequantization_on_the_cpu_oracle() {
    check_against_f16_copy("IQ1_M", Engine::Oracle, ORACLE_WINDOWS);
}

#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; loads a 0.5B IQ1 GGUF; the checkpoint-rocm lane runs it by name"]
fn iq1_s_matches_llamacpp_running_its_f16_dequantization_on_rocm() {
    check_against_f16_copy(
        "IQ1_S",
        Engine::Device(BackendChoice::Rocm, DeviceBackend::Rocm),
        DEVICE_WINDOWS,
    );
}

#[test]
#[ignore = "needs the ROCm device and /tmp/poot-gpu.lock; loads a 0.5B IQ1 GGUF; the checkpoint-rocm lane runs it by name"]
fn iq1_m_matches_llamacpp_running_its_f16_dequantization_on_rocm() {
    check_against_f16_copy(
        "IQ1_M",
        Engine::Device(BackendChoice::Rocm, DeviceBackend::Rocm),
        DEVICE_WINDOWS,
    );
}

/// The control: poot scores the IQ1_S file against llama.cpp's reference of the IQ1_M file. A different
/// quantization of the same model is a decode that is wrong in its values, and the f16 gate sees it.
#[test]
fn a_different_quantization_leaves_the_f16_gate_on_the_cpu_oracle() {
    let (Some(s), Some(m_copy)) = (checkpoint("IQ1_S"), f16_copy("IQ1_M")) else {
        return;
    };
    let Some(agreement) = measure(&m_copy, &s, Engine::Oracle, ORACLE_WINDOWS) else {
        return;
    };
    eprintln!("IQ1_S scored against the IQ1_M f16 copy: {agreement:?}");
    assert!(
        F16_GATE.check(&agreement).is_err(),
        "the f16 gate passed a different quantization: {agreement:?}"
    );
}
