//! POOT-1141: a teacher-forced gate against llama.cpp on the same GGUF.
//!
//! llama.cpp scores a fixed token stream (`llama-perplexity --save-all-logits`) and poot scores the same
//! stream chunk by chunk with the logits at every position, through the driver's own trace, packed-weight
//! binding, `compile_staged` and executor. At each scored position the two next-token distributions are
//! compared: KL(llama.cpp || poot), the difference of the two negative log-likelihoods of the token that
//! actually follows, and whether the two arg-maxes agree. A [`Gate`] holds the thresholds.
//!
//! The reference is a pinned binary, never a build of a downloaded tree: `llama-perplexity` from the Nix
//! package, at [`LLAMA_CPP_COMMIT`], run on the CPU with the flags in [`reference_flags`]. A different
//! commit fails the row (the thresholds were measured against this one) rather than skipping it.
//!
//! Small models only (a 0.5B Q8_0 and Q4_K_M GGUF); a 20B+ checkpoint never runs locally.
//!
//! Layout of the reference file (llama.cpp `tools/perplexity/perplexity.cpp`, `perplexity()` and
//! `kl_divergence()`): the magic `_logits_`; `n_ctx`, `n_vocab` and `n_chunk` as little-endian 32-bit
//! integers; the `n_chunk * n_ctx` tokens as 32-bit integers; then, for each chunk, one record for each of
//! the positions `n_ctx / 2 .. n_ctx - 1` (the second half of the window, so the model has context). A
//! record is `2 * ((n_vocab + 1) / 2) + 4` 16-bit values: an `f32` scale and an `f32` log-probability
//! floor, then one 16-bit code per vocabulary entry; entry `i` has log-probability `scale * code[i] + floor`.
//! The record at position `p` is the distribution of the token at `p + 1`.

use std::num::NonZeroUsize;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use poot_executor::{EntryId, ExecutableId, Executor, StateScope};
use poot_graph_plan::{
    CompileLimits, CompileOptions, WeightFormats, bind_packed_weights,
    multi_device::topology::DeviceId,
    staged::{DevicePlacement, ExpertPlacement, Partition, compile_staged},
};
use poot_models::model::{LogitRows, StepShape, StepSpec};
use poot_models::test_support::weight_map::Step;

use super::{BackendChoice, ModelHandle, open_executor, serving_trace_limits};

/// The llama.cpp commit the thresholds below were measured against (`llama-perplexity --version`).
const LLAMA_CPP_COMMIT: &str = "d812350";

/// The text both engines score: prose and one line of code, five windows of [`N_CTX`] tokens.
const CORPUS: &str = include_str!("../../tests/fixtures/kl_corpus.txt");

/// The window llama.cpp scores in: 128 tokens, the second half scored.
const N_CTX: usize = 128;

/// The flags every reference run uses, after `-m <model> -f <corpus> --save-all-logits <out>`: CPU only
/// (`-dev none`, no offload) so the reference does not depend on a GPU backend, no warmup run, one window per
/// batch.
fn reference_flags() -> Vec<String> {
    let n_ctx = N_CTX.to_string();
    [
        "-c",
        &n_ctx,
        "-b",
        &n_ctx,
        "-ngl",
        "0",
        "-dev",
        "none",
        "-t",
        "8",
        "--no-warmup",
    ]
    .map(String::from)
    .to_vec()
}

/// A malformed llama.cpp logits file.
#[derive(Debug, thiserror::Error, PartialEq)]
enum ReferenceError {
    #[error("not a llama.cpp logits file (magic {0:?})")]
    Magic(Vec<u8>),
    #[error("the file ends inside {0}")]
    Truncated(&'static str),
    #[error("{0} is {1}, which a logits file never holds")]
    Header(&'static str, i64),
}

/// A llama.cpp `--save-all-logits` file, read in place.
struct Reference<'a> {
    n_ctx: usize,
    n_vocab: usize,
    chunks: usize,
    tokens: Vec<i32>,
    /// The records, back to back: `chunks * scored()` of `record_u16s()` values each.
    records: &'a [u8],
}

impl<'a> Reference<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, ReferenceError> {
        const MAGIC: &[u8] = b"_logits_";
        let head = bytes
            .get(..MAGIC.len())
            .ok_or(ReferenceError::Truncated("the magic"))?;
        if head != MAGIC {
            return Err(ReferenceError::Magic(head.to_vec()));
        }
        let mut at = MAGIC.len();
        let mut int = |what: &'static str| -> Result<i64, ReferenceError> {
            let raw = bytes
                .get(at..at + 4)
                .ok_or(ReferenceError::Truncated(what))?;
            at += 4;
            Ok(i64::from(i32::from_le_bytes(raw.try_into().unwrap())))
        };
        let (n_ctx, n_vocab, chunks) = (int("n_ctx")?, int("n_vocab")?, int("n_chunk")?);
        // A window needs a scored position (`n_ctx - 1 - n_ctx / 2 >= 1`), a vocabulary and a window.
        for (what, value, least) in [
            ("n_ctx", n_ctx, 3),
            ("n_vocab", n_vocab, 1),
            ("n_chunk", chunks, 1),
        ] {
            if value < least {
                return Err(ReferenceError::Header(what, value));
            }
        }
        let (n_ctx, n_vocab, chunks) = (n_ctx as usize, n_vocab as usize, chunks as usize);
        let token_bytes = bytes
            .get(at..at + 4 * n_ctx * chunks)
            .ok_or(ReferenceError::Truncated("the tokens"))?;
        let tokens = token_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        at += token_bytes.len();
        let reference = Self {
            n_ctx,
            n_vocab,
            chunks,
            tokens,
            records: &[],
        };
        let size = 2 * reference.record_u16s() * reference.scored() * chunks;
        let records = bytes
            .get(at..at + size)
            .ok_or(ReferenceError::Truncated("the records"))?;
        Ok(Self {
            records,
            ..reference
        })
    }

    /// The first scored position of a window.
    fn first(&self) -> usize {
        self.n_ctx / 2
    }

    /// Scored positions per window.
    fn scored(&self) -> usize {
        self.n_ctx - 1 - self.first()
    }

    fn record_u16s(&self) -> usize {
        2 * self.n_vocab.div_ceil(2) + 4
    }

    fn window_tokens(&self, chunk: usize) -> &[i32] {
        &self.tokens[chunk * self.n_ctx..][..self.n_ctx]
    }

    /// The log-probabilities llama.cpp gave the token after scored position `j` of `chunk`, with
    /// `NEG_INFINITY` for a token the file does not represent.
    ///
    /// The writer clamps a log-probability to `max - 16` nats and spreads 65535 codes over the range above
    /// it, so every token below the clamp gets code 0, which decodes to the floor: a probability the model
    /// never gave. With a 152k-entry vocabulary 129k tokens sit below the clamp at a typical position, and
    /// their floors read as probabilities add up to 0.6% of the mass. Code 0 therefore means "below the
    /// clamp", not a value.
    fn log_probs(&self, chunk: usize, j: usize) -> Vec<f64> {
        let at = 2 * self.record_u16s() * (chunk * self.scored() + j);
        let record = &self.records[at..][..2 * self.record_u16s()];
        let float =
            |i: usize| f64::from(f32::from_le_bytes(record[4 * i..][..4].try_into().unwrap()));
        let (scale, floor) = (float(0), float(1));
        record[8..][..2 * self.n_vocab]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| match u16::from_le_bytes(*b) {
                0 => f64::NEG_INFINITY,
                code => scale * f64::from(code) + floor,
            })
            .collect()
    }
}

/// `log_softmax` of `logits`, in f64.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
    let sum: f64 = logits.iter().map(|&x| f64::from(x - max).exp()).sum();
    let log_sum = sum.ln() + f64::from(max);
    logits.iter().map(|&x| f64::from(x) - log_sum).collect()
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0
}

/// How closely poot's next-token distributions follow llama.cpp's over every scored position.
#[derive(Clone, Debug, PartialEq)]
struct Agreement {
    positions: usize,
    /// Mean of KL(llama.cpp || poot), in nats.
    mean_kl: f64,
    /// Largest KL at one position.
    max_kl: f64,
    /// Mean negative log-likelihood of the followed token under poot, minus llama.cpp's.
    nll_delta: f64,
    /// Fraction of positions where both engines rank the same token first.
    top1: f64,
}

/// Compare `reference` with the logits `poot` returns for a window's tokens (`n_ctx * n_vocab` values, one
/// row per position).
fn compare(reference: &Reference<'_>, mut poot: impl FnMut(&[i32]) -> Vec<f32>) -> Agreement {
    let (mut kl_sum, mut kl_max, mut nll_delta, mut nll_positions, mut agree, mut positions) =
        (0.0, 0.0f64, 0.0, 0usize, 0, 0);
    for chunk in 0..reference.chunks {
        let window = reference.window_tokens(chunk);
        let logits = poot(window);
        assert_eq!(
            logits.len(),
            reference.n_ctx * reference.n_vocab,
            "poot returns one row of {} logits per position",
            reference.n_vocab
        );
        for j in 0..reference.scored() {
            let position = reference.first() + j;
            let next = window[position + 1] as usize;
            // Both distributions are compared over the tokens the reference represents, each renormalized
            // there. Reading the clamped tail as floor-sized probabilities only the reference holds put the
            // mean KL of llama.cpp against its own file at 0.025 (see `Reference::log_probs`).
            let mut base = reference.log_probs(chunk, j);
            let mut ours =
                log_softmax(&logits[position * reference.n_vocab..][..reference.n_vocab]);
            for (o, b) in ours.iter_mut().zip(&base) {
                if *b == f64::NEG_INFINITY {
                    *o = f64::NEG_INFINITY;
                }
            }
            for lp in [&mut base, &mut ours] {
                let mass: f64 = lp.iter().map(|x| x.exp()).sum();
                lp.iter_mut().for_each(|x| *x -= mass.ln());
            }
            let kl: f64 = base
                .iter()
                .zip(&ours)
                .filter(|&(&b, _)| b > f64::NEG_INFINITY)
                .map(|(&b, &o)| b.exp() * (b - o))
                .sum();
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            // A followed token below the clamp has no reference value: it adds no NLL term.
            if base[next] > f64::NEG_INFINITY {
                nll_delta += base[next] - ours[next];
                nll_positions += 1;
            }
            agree += usize::from(argmax(&base) == argmax(&ours));
            positions += 1;
        }
    }
    let n = positions as f64;
    Agreement {
        positions,
        mean_kl: kl_sum / n,
        max_kl: kl_max,
        nll_delta: nll_delta / nll_positions as f64,
        top1: agree as f64 / n,
    }
}

/// The thresholds an [`Agreement`] must meet.
#[derive(Clone, Copy, Debug)]
struct Gate {
    max_mean_kl: f64,
    max_max_kl: f64,
    max_abs_nll_delta: f64,
    min_top1: f64,
}

/// A threshold an [`Agreement`] missed.
#[derive(Debug, thiserror::Error, PartialEq)]
enum GateFailure {
    #[error("mean KL {got:.6} nats exceeds the gate {limit:.6}")]
    MeanKl { got: f64, limit: f64 },
    #[error("max KL {got:.6} nats exceeds the gate {limit:.6}")]
    MaxKl { got: f64, limit: f64 },
    #[error("NLL differs from llama.cpp's by {got:.6} nats, beyond the gate {limit:.6}")]
    NllDelta { got: f64, limit: f64 },
    #[error("top-1 agreement {got:.4} is below the gate {limit:.4}")]
    Top1 { got: f64, limit: f64 },
}

impl Gate {
    fn check(&self, a: &Agreement) -> Result<(), GateFailure> {
        // A NaN measurement misses every threshold.
        let above = |got: f64, limit: f64| got.is_nan() || got > limit;
        let below = |got: f64, limit: f64| got.is_nan() || got < limit;
        if above(a.mean_kl, self.max_mean_kl) {
            return Err(GateFailure::MeanKl {
                got: a.mean_kl,
                limit: self.max_mean_kl,
            });
        }
        if above(a.max_kl, self.max_max_kl) {
            return Err(GateFailure::MaxKl {
                got: a.max_kl,
                limit: self.max_max_kl,
            });
        }
        if above(a.nll_delta.abs(), self.max_abs_nll_delta) {
            return Err(GateFailure::NllDelta {
                got: a.nll_delta,
                limit: self.max_abs_nll_delta,
            });
        }
        if below(a.top1, self.min_top1) {
            return Err(GateFailure::Top1 {
                got: a.top1,
                limit: self.min_top1,
            });
        }
        Ok(())
    }
}

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

/// poot's logits for whole windows: one prefill program with logits at every row, replayed per window.
struct TeacherForced {
    exec: Box<dyn Executor>,
    exe: ExecutableId,
    entry: EntryId,
    n_ctx: usize,
}

impl TeacherForced {
    fn new(handle: &ModelHandle, mut exec: Box<dyn Executor>, n_ctx: usize) -> Self {
        let model = handle.model();
        let shape = StepShape::new(nz(1), nz(n_ctx), nz(n_ctx)).with_logits(LogitRows::All);
        let graph = model
            .trace(&StepSpec::step(shape), serving_trace_limits(nz(n_ctx)))
            .expect("trace the teacher-forced prefill");
        let graph = bind_packed_weights(&graph, &WeightFormats::from_weight_map(model.weights()))
            .expect("bind the packed weights");
        let program = compile_staged(
            &graph,
            &exec.target_set(),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions::new(CompileLimits::STANDARD),
        )
        .expect("compile the teacher-forced prefill");
        let exe = exec
            .load_weights(
                Arc::clone(handle.shared_store()),
                Arc::new(model.weights().clone()),
            )
            .expect("load the weights");
        let entry = exec.add_entry(exe, &program).expect("add the entry");
        Self {
            exec,
            exe,
            entry,
            n_ctx,
        }
    }

    /// The logits of every position of `window`, `[n_ctx, n_vocab]` row-major.
    fn logits(&mut self, window: &[i32]) -> Vec<f32> {
        assert_eq!(window.len(), self.n_ctx);
        self.exec
            .reset_state(self.exe, StateScope::All)
            .expect("reset the carried state");
        let step = Step::new(window, 0);
        let out = self
            .exec
            .step(
                self.exe,
                self.entry,
                &step.inputs(),
                &mut poot_executor::NoSync,
            )
            .expect("step")
            .to_host()
            .expect("read the logits");
        out.as_f32().expect("f32 logits").to_vec()
    }
}

impl Drop for TeacherForced {
    fn drop(&mut self) {
        let _ = self.exec.unload(self.exe);
    }
}

/// The reference's tokens, scored by `llama-perplexity` on `model` and `corpus`, written to `out`. `None`
/// (after saying why) when the binary is not on `PATH`.
fn run_reference(model: &Path, corpus: &Path, out: &Path) -> Option<()> {
    let version = match Command::new("llama-perplexity").arg("--version").output() {
        Ok(output) => output,
        Err(error) => {
            eprintln!("SKIP: llama-perplexity is not on PATH ({error})");
            return None;
        }
    };
    let version = String::from_utf8_lossy(&version.stderr).into_owned()
        + &String::from_utf8_lossy(&version.stdout);
    assert!(
        version.contains(&format!("commit {LLAMA_CPP_COMMIT}")),
        "llama-perplexity is not the pinned llama.cpp commit {LLAMA_CPP_COMMIT}: {version}; re-measure the gates"
    );
    let run = Command::new("llama-perplexity")
        .arg("-m")
        .arg(model)
        .arg("-f")
        .arg(corpus)
        .args(reference_flags())
        .arg("--save-all-logits")
        .arg(out)
        .output()
        .expect("run llama-perplexity");
    assert!(
        run.status.success(),
        "llama-perplexity failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    Some(())
}

// ---------------------------------------------------------------------------------------------
// Model-free: the file format, the metrics and the gate, on distributions built in the test.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod format_tests {
    use super::*;

    /// llama.cpp's `log_softmax` quantizer (`perplexity.cpp`), the writer the reader inverts.
    fn record(logits: &[f32]) -> Vec<u16> {
        let max = logits.iter().fold(f32::MIN, |m, &x| m.max(x));
        let min = logits
            .iter()
            .fold(f32::MAX, |m, &x| m.min(x))
            .max(max - 16.0);
        let sum: f64 = logits.iter().map(|&x| f64::from(x - max).exp()).sum();
        let floor = min - max - sum.ln() as f32;
        let scale = (max - min) / 65535.0;
        let mut out: Vec<u16> = [scale, floor]
            .iter()
            .flat_map(|f| {
                let b = f.to_le_bytes();
                [
                    u16::from_le_bytes([b[0], b[1]]),
                    u16::from_le_bytes([b[2], b[3]]),
                ]
            })
            .collect();
        out.extend(logits.iter().map(|&x| {
            if scale > 0.0 && x > min {
                ((x - min) / scale).round() as u16
            } else {
                0
            }
        }));
        out.resize(2 * logits.len().div_ceil(2) + 4, 0);
        out
    }

    /// A reference file whose window `c` has the tokens `tokens[c]` and whose scored position `j`
    /// carries `logits(c, j)`.
    pub(super) fn file(
        n_ctx: usize,
        n_vocab: usize,
        tokens: &[Vec<i32>],
        logits: impl Fn(usize, usize) -> Vec<f32>,
    ) -> Vec<u8> {
        let mut out = b"_logits_".to_vec();
        for n in [n_ctx, n_vocab, tokens.len()] {
            out.extend((n as i32).to_le_bytes());
        }
        out.extend(tokens.iter().flatten().flat_map(|t| t.to_le_bytes()));
        for c in 0..tokens.len() {
            for j in 0..n_ctx - 1 - n_ctx / 2 {
                out.extend(record(&logits(c, j)).iter().flat_map(|v| v.to_le_bytes()));
            }
        }
        out
    }

    #[test]
    fn a_reference_file_reads_back_its_tokens_and_its_distributions() {
        let tokens = vec![vec![3, 1, 4, 1, 5, 9], vec![2, 6, 5, 3, 5, 8]];
        let logits = |c: usize, j: usize| vec![0.5 * c as f32, 2.0 + j as f32, -1.0, 0.25, 3.0];
        let bytes = file(6, 5, &tokens, logits);
        let reference = Reference::parse(&bytes).unwrap();
        assert_eq!(
            (reference.n_ctx, reference.n_vocab, reference.chunks),
            (6, 5, 2)
        );
        assert_eq!(reference.window_tokens(1), [2, 6, 5, 3, 5, 8]);
        assert_eq!((reference.first(), reference.scored()), (3, 2));
        for (c, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let want = log_softmax(&logits(c, j));
            let got = reference.log_probs(c, j);
            // Codes are 16 bits over a range of at most 16 nats. The smallest logit takes code 0, which
            // means "not represented", every other entry reads back.
            let smallest = want.iter().cloned().fold(f64::INFINITY, f64::min);
            for (g, w) in got.iter().zip(&want) {
                if *w == smallest {
                    assert_eq!(*g, f64::NEG_INFINITY);
                } else {
                    assert!((g - w).abs() < 1e-3, "{g} vs {w}");
                }
            }
        }
    }

    #[test]
    fn a_malformed_reference_file_is_a_typed_error() {
        let bytes = file(6, 5, &[vec![0; 6]], |_, _| vec![0.0; 5]);
        assert!(matches!(
            Reference::parse(b"_logitz_\0\0\0\0"),
            Err(ReferenceError::Magic(_))
        ));
        assert_eq!(
            Reference::parse(&bytes[..bytes.len() - 1]).err(),
            Some(ReferenceError::Truncated("the records"))
        );
        assert_eq!(
            Reference::parse(&bytes[..20]).err(),
            Some(ReferenceError::Truncated("the tokens"))
        );
        let mut zero_vocab = bytes.clone();
        zero_vocab[12..16].copy_from_slice(&0i32.to_le_bytes());
        assert_eq!(
            Reference::parse(&zero_vocab).err(),
            Some(ReferenceError::Header("n_vocab", 0))
        );
    }
}

#[cfg(test)]
mod gate_tests {
    use super::format_tests::file;
    use super::*;

    const N_CTX: usize = 8;
    const VOCAB: usize = 40;

    /// Window tokens that cycle through the vocabulary.
    fn tokens(chunks: usize) -> Vec<Vec<i32>> {
        (0..chunks)
            .map(|c| (0..N_CTX).map(|i| ((3 * i + c) % VOCAB) as i32).collect())
            .collect()
    }

    /// Reference logits with one clear winner per position and a graded tail.
    fn logits(chunk: usize, j: usize) -> Vec<f32> {
        (0..VOCAB)
            .map(|v| {
                let winner = (7 * chunk + 5 * j) % VOCAB;
                if v == winner {
                    6.0
                } else {
                    ((v * 13 + j) % 11) as f32 * 0.3
                }
            })
            .collect()
    }

    fn reference_bytes() -> Vec<u8> {
        file(N_CTX, VOCAB, &tokens(3), logits)
    }

    /// poot's logits when it reproduces the reference's, passed through `edit` per (position, row).
    fn poot(
        reference: &Reference<'_>,
        edit: impl Fn(usize, &mut [f32]),
    ) -> impl FnMut(&[i32]) -> Vec<f32> {
        let windows: Vec<Vec<i32>> = (0..reference.chunks)
            .map(|c| reference.window_tokens(c).to_vec())
            .collect();
        let first = reference.first();
        move |window: &[i32]| {
            let chunk = windows.iter().position(|w| w == window).unwrap();
            let mut out = Vec::new();
            for position in 0..N_CTX {
                let mut row = logits(chunk, position.saturating_sub(first));
                edit(position, &mut row);
                out.extend(row);
            }
            out
        }
    }

    const GATE: Gate = Gate {
        max_mean_kl: 1e-3,
        max_max_kl: 5e-3,
        max_abs_nll_delta: 1e-3,
        min_top1: 0.99,
    };

    #[test]
    fn poot_reproducing_the_reference_passes_the_gate() {
        let bytes = reference_bytes();
        let reference = Reference::parse(&bytes).unwrap();
        let agreement = compare(&reference, poot(&reference, |_, _| {}));
        assert_eq!(agreement.positions, 9);
        assert!(agreement.mean_kl < 1e-5, "{agreement:?}");
        assert_eq!(agreement.top1, 1.0);
        assert_eq!(GATE.check(&agreement), Ok(()));
    }

    /// Mutation: skip the `nll_delta` check in `Gate::check`; the shifted row stops failing.
    #[test]
    fn a_perturbed_distribution_fails_each_threshold() {
        let bytes = reference_bytes();
        let reference = Reference::parse(&bytes).unwrap();
        // One logit raised by a scale error: the winner's rival overtakes it.
        let overtaken = compare(
            &reference,
            poot(&reference, |_, row| {
                let winner = argmax(&row.iter().map(|&x| f64::from(x)).collect::<Vec<_>>());
                row[(winner + 1) % VOCAB] += 4.0;
            }),
        );
        assert!(overtaken.top1 < GATE.min_top1, "{overtaken:?}");
        assert!(matches!(
            GATE.check(&overtaken),
            Err(GateFailure::MeanKl { .. })
        ));
        // The same shift alone, with the other limits relaxed, is caught by each in turn.
        let loose = Gate {
            max_mean_kl: f64::INFINITY,
            max_max_kl: f64::INFINITY,
            max_abs_nll_delta: f64::INFINITY,
            min_top1: 0.0,
        };
        for (gate, failure) in [
            (
                Gate {
                    max_mean_kl: 1e-3,
                    ..loose
                },
                "MeanKl",
            ),
            (
                Gate {
                    max_max_kl: 1e-3,
                    ..loose
                },
                "MaxKl",
            ),
            (
                Gate {
                    min_top1: 0.99,
                    ..loose
                },
                "Top1",
            ),
        ] {
            let name = format!("{:?}", gate.check(&overtaken).unwrap_err());
            assert!(name.starts_with(failure), "{name}");
        }
        // A uniform shift of the next token's logit moves its NLL and nothing else much.
        let nll_only = Gate {
            max_abs_nll_delta: 1e-3,
            ..loose
        };
        let shifted = compare(
            &reference,
            poot(&reference, |_, row| row.iter_mut().for_each(|x| *x *= 1.2)),
        );
        assert!(
            matches!(nll_only.check(&shifted), Err(GateFailure::NllDelta { .. })),
            "{shifted:?}"
        );
    }

    /// POOT-1156: a large vocabulary whose tail the file clamps. poot reproduces the reference's logits
    /// exactly, so the agreement is perfect; the tail's floor, read as probability, is 1.1% of the mass
    /// (100k tokens at `e^-16` of the winner's) and must not show up as KL.
    ///
    /// Mutation: drop the `NEG_INFINITY` arm of `Reference::log_probs` (decode code 0 as the floor); the
    /// mean KL becomes ~0.1 and this test is red.
    #[test]
    fn a_clamped_tail_is_not_read_as_probability() {
        const WIDE: usize = 100_000;
        const WINDOW: usize = 8;
        // Three live tokens, the rest 30 nats below: far under the `max - 16` clamp.
        let wide = |winner: usize| -> Vec<f32> {
            (0..WIDE)
                .map(|v| match (v + WIDE - winner) % WIDE {
                    0 => 8.0,
                    1 => 7.0,
                    2 => 6.0,
                    _ => -22.0,
                })
                .collect()
        };
        let tokens = vec![(0..WINDOW as i32).collect::<Vec<_>>()];
        let bytes = file(WINDOW, WIDE, &tokens, |_, j| wide(5 * j));
        let reference = Reference::parse(&bytes).unwrap();
        let agreement = compare(&reference, |_| {
            (0..WINDOW)
                .flat_map(|position| wide(5 * position.saturating_sub(reference.first())))
                .collect()
        });
        assert_eq!(agreement.positions, 3);
        assert!(agreement.mean_kl < 1e-4, "{agreement:?}");
        assert_eq!(agreement.top1, 1.0);
    }

    #[test]
    fn a_nan_logit_fails_the_gate() {
        let bytes = reference_bytes();
        let reference = Reference::parse(&bytes).unwrap();
        let agreement = compare(&reference, poot(&reference, |_, row| row[0] = f32::NAN));
        assert!(GATE.check(&agreement).is_err(), "{agreement:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Checkpoint rows: the real GGUFs under `POOT_MODELS_DIR`, scored by llama.cpp and by poot on a device.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod checkpoint_rows {
    use std::io::{Read, Seek, SeekFrom, Write};

    use super::*;
    use poot_load::gguf::GgufIndex;
    use poot_models::registry::Registry;
    use poot_quant::StoredBytes;
    use poot_quant::scalar::{f16_to_f32, f32_to_f16};
    use poot_runtime_common::DeviceBackend;

    /// What poot measured against llama.cpp `d812350` on the wgpu device (RADV, Strix Halo), over [`CORPUS`]
    /// (315 scored positions), compared over the tokens the reference file represents (see
    /// `Reference::log_probs`):
    ///
    /// | model                        | mean KL | max KL | NLL delta | top-1 |
    /// |------------------------------|---------|--------|-----------|-------|
    /// | Qwen2.5-0.5B-Instruct Q8_0   | 0.0015  | 0.0093 | -0.0022   | 0.956 |
    /// | Qwen2.5-0.5B-Instruct Q4_K_M | 0.0023  | 0.0268 | -0.0099   | 0.949 |
    /// | Qwen3-0.6B Q8_0              | 0.0015  | 0.0088 | +0.0061   | 0.975 |
    ///
    /// That is llama.cpp's own noise: 0.0014 between its CPU and Vulkan backends, 0.0015 against an f16 copy
    /// of the weights. The 0.032 mean KL this gate first recorded (POOT-1141) was the reference file's
    /// clamped tail read as probability, not poot: llama.cpp's own logits scored 0.025 against its own
    /// file. POOT-1156 bisected per layer (`llamacpp_bisect_tests`): poot's CPU oracle matches llama.cpp
    /// running an f16 copy of the weights to 0.18% at the last layer and 0.14% at the logits (0.00000 mean KL
    /// on the 128-token window), and
    /// differs from llama.cpp's Q8_0 run by the Q8_0 quantization of its activations, which the f16 copy
    /// does not have. The gate holds the measured state with 30% room on the mean KL.
    const GATE: Gate = Gate {
        max_mean_kl: 0.003,
        max_max_kl: 0.04,
        max_abs_nll_delta: 0.015,
        min_top1: 0.92,
    };

    fn load(gguf: &Path) -> ModelHandle {
        ModelHandle::load(gguf, &Registry::builtin().unwrap(), &Default::default())
            .expect("load the GGUF")
    }

    /// The agreement of `scored` (loaded and run by poot on `choice`) with llama.cpp's reference over
    /// [`CORPUS`] computed on `reference`; `None` when the device or the reference tool is unavailable.
    /// The two are the same file except in the mutation row.
    fn measure(
        reference: &Path,
        scored: &Path,
        choice: BackendChoice,
        backend: DeviceBackend,
    ) -> Option<Agreement> {
        let exec = poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice))?;
        let scratch = poot_test_util::unique_temp_path("reference.bin");
        let corpus = scratch.with_file_name("corpus.txt");
        std::fs::write(&corpus, CORPUS).unwrap();
        run_reference(reference, &corpus, &scratch)?;
        let bytes = std::fs::read(&scratch).unwrap();
        let reference = Reference::parse(&bytes).expect("parse the reference");
        assert_eq!(reference.n_ctx, N_CTX);
        let handle = load(scored);
        assert_eq!(
            handle.config().vocab,
            reference.n_vocab,
            "the two engines share one vocabulary"
        );
        let mut forward = TeacherForced::new(&handle, exec, N_CTX);
        Some(compare(&reference, |window| forward.logits(window)))
    }

    fn check(gguf: &Path) {
        let Some(agreement) = measure(gguf, gguf, BackendChoice::Wgpu, DeviceBackend::Wgpu) else {
            return;
        };
        eprintln!("llama.cpp agreement: {agreement:?}");
        assert_eq!(agreement.positions, 5 * (N_CTX - 1 - N_CTX / 2));
        if let Err(failure) = GATE.check(&agreement) {
            panic!("{}: {failure} ({agreement:?})", gguf.display());
        }
    }

    #[test]
    fn qwen2_5_0_5b_q8_0_follows_llamacpp() {
        let Some(gguf) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q8_0.gguf"
        )) else {
            return;
        };
        check(&gguf);
    }

    #[test]
    fn qwen2_5_0_5b_q4_k_m_follows_llamacpp() {
        let Some(gguf) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        )) else {
            return;
        };
        check(&gguf);
    }

    #[test]
    fn qwen3_0_6b_q8_0_follows_llamacpp() {
        let Some(gguf) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "qwen3-0.6b-gguf/Qwen3-0.6B-Q8_0.gguf"
        )) else {
            return;
        };
        check(&gguf);
    }

    /// A copy of the Q8_0 GGUF `source` at `target` whose tensor `name` has the f16 scale of block `block`
    /// multiplied by `factor`: one dequantized scale, 32 weights, changed.
    fn copy_with_scaled_block(source: &Path, target: &Path, name: &str, block: usize, factor: f32) {
        const Q8_0: u32 = 8;
        const F32: u32 = 0;
        const Q8_0_BLOCK_BYTES: u64 = 34;
        std::fs::copy(source, target).expect("copy the GGUF");
        let file = std::fs::File::open(source).unwrap();
        let length = file.metadata().unwrap().len();
        let index = GgufIndex::parse(&StoredBytes::map(file).unwrap()).unwrap();
        let bytes_of = |dims: &[u64], ggml_type: u32| -> u64 {
            let elements: u64 = dims.iter().product();
            match ggml_type {
                F32 => 4 * elements,
                Q8_0 => elements / 32 * Q8_0_BLOCK_BYTES,
                other => panic!("this helper sizes F32 and Q8_0 tensors, not ggml type {other}"),
            }
        };
        // Offsets are relative to the tensor-data section, which runs to the end of the file: its start
        // is the file length less the end of the last tensor.
        let data_end = index
            .tensors
            .values()
            .map(|t| t.offset + bytes_of(&t.dims, t.ggml_type))
            .max()
            .unwrap();
        let data_start = length - data_end;
        assert_eq!(
            data_start % 32,
            0,
            "the tensor data starts on the alignment"
        );
        let tensor = &index.tensors[name];
        assert_eq!(tensor.ggml_type, Q8_0, "{name} is a Q8_0 tensor");
        let at = data_start + tensor.offset + Q8_0_BLOCK_BYTES * block as u64;
        assert!(at + 2 <= data_start + tensor.offset + bytes_of(&tensor.dims, Q8_0));
        let mut copy = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(target)
            .unwrap();
        let mut scale = [0u8; 2];
        copy.seek(SeekFrom::Start(at)).unwrap();
        copy.read_exact(&mut scale).unwrap();
        let old = f16_to_f32(u16::from_le_bytes(scale));
        assert!(
            old.is_finite() && old > 0.0,
            "block {block} of {name} reads scale {old}"
        );
        copy.seek(SeekFrom::Start(at)).unwrap();
        copy.write_all(&f32_to_f16(old * factor).to_le_bytes())
            .unwrap();
    }

    /// The gate has teeth: one dequantization scale (block 13 of the output head's row for the token
    /// ` the`, 32 of its 896 weights) multiplied by -4 in a copy of the weights takes the run over the gate
    /// against llama.cpp's reference of the intact file, while the intact file passes it.
    ///
    /// Mutation: leave the copy unedited in `copy_with_scaled_block` (drop its final `write_all`); the
    /// perturbed run equals the intact one and `expect_err` below is red.
    #[test]
    fn one_perturbed_dequant_scale_exceeds_the_gate() {
        let Some(gguf) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q8_0.gguf"
        )) else {
            return;
        };
        // The output head is `[vocab, 896]`: 28 blocks of 32 weights per row.
        const BLOCKS_PER_ROW: usize = 896 / 32;
        const THE: usize = 279;
        let scratch = poot_test_util::unique_temp_path("perturbed.gguf");
        copy_with_scaled_block(
            &gguf,
            &scratch,
            "output.weight",
            THE * BLOCKS_PER_ROW + 13,
            -4.0,
        );
        let Some(intact) = measure(&gguf, &gguf, BackendChoice::Wgpu, DeviceBackend::Wgpu) else {
            return;
        };
        let perturbed = measure(&gguf, &scratch, BackendChoice::Wgpu, DeviceBackend::Wgpu)
            .expect("the device opened a moment ago");
        eprintln!("intact {intact:?}\nperturbed {perturbed:?}");
        assert_eq!(GATE.check(&intact), Ok(()));
        let failure = GATE
            .check(&perturbed)
            .expect_err("one perturbed scale must fail the gate");
        eprintln!("the gate refuses the perturbed copy: {failure}");
    }
}
