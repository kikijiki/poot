//! Criterion perf benches for the CPU-side decode and prefill pipelines (perf-regression suite, Stage B).
//! Covers the hot host paths: tracing the graph, the cse + G5 fusion pass, and one CPU eager forward
//! (`eval_with_state`) un-fused vs fused, for the masked decode step (one token, the per-token hot path) and
//! the whole-prompt prefill (`Model::trace` of a prefill step, the TTFT hot path, at two prompt lengths). Larger configs
//! keep timings off the fixed graph-walk floor. GPU/PTX perf is measured by the RunPod cross-framework sweep.
//!
//! Run: `cargo bench -p poot-eval` (inside `nix develop`).
//!
//! Criterion compares against the last run and prints `change: [-x% +y%]`. To pin a named baseline:
//!   `cargo bench -p poot-eval -- --save-baseline main`   (record once, e.g. on master)
//!   `cargo bench -p poot-eval -- --baseline main`        (compare a change against it)
//! A regression shows as a positive `change` percentage with a `Performance has regressed` note.

use std::collections::HashMap;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use poot_eval::{EvalBudget, EvalOptions, Value};
use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_executor_parity::weight_map::MappedModel;
use poot_graph_ir::{Graph, Slot, Storage};
use poot_models::model::{LogitRows, Phase};
use poot_tensor::HostTensor;

use poot_graph_plan as transform;

/// The small qwen2 (through the registry): 4 layers, hidden 64, GQA 8:4, head_dim 8.
fn model() -> MappedModel {
    Dense::new(Family::Qwen2)
        .vocab(256)
        .dims(64, 128, 4)
        .heads(8, 4)
        .head_dim(8)
        .max_positions(64)
        .f32_model()
}

/// `m`'s step graph: one sequence of `tokens` new tokens over `cap` positions.
fn trace(
    m: &MappedModel,
    phase: Phase,
    tokens: usize,
    cap: usize,
) -> poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs> {
    m.model
        .trace(phase, step(1, tokens, cap, LogitRows::Last))
        .unwrap()
}

fn weight(name: &str, shape: &[usize]) -> HostTensor {
    let seed: u64 = name.bytes().fold(1469598103934665603u64, |h, c| {
        (h ^ c as u64).wrapping_mul(1099511628211)
    });
    let mut s = seed | 1;
    let data: Vec<f32> = (0..shape.iter().product::<usize>().max(1))
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32) / ((1u64 << 23) as f32) - 1.0) * 0.1
        })
        .collect();
    HostTensor::f32(shape.to_vec(), data)
}

/// Bind a step graph for timing: token ids `i % 17` and positions `0..tokens` (any in-vocab ids and
/// in-range positions work), every `Const` filled by [`weight`], caches seeded to zeros. The step has no
/// other slots.
fn bind(g: &Graph) -> HashMap<usize, Value> {
    let mut inputs: HashMap<usize, Value> = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        let numel = m.aval.shape.iter().product::<usize>().max(1);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                inputs.insert(
                    id,
                    Value::from(HostTensor::i32(
                        m.aval.shape.clone(),
                        (0..numel).map(|i| (i % 17) as i32).collect(),
                    )),
                );
            }
            Storage::Slot(Slot::Pos) => {
                inputs.insert(
                    id,
                    Value::from(HostTensor::i32(
                        m.aval.shape.clone(),
                        (0..numel as i32).collect(),
                    )),
                );
            }
            Storage::Slot(other) => unreachable!("unexpected slot {other:?} in a dense step graph"),
            Storage::Const => {
                inputs.insert(
                    id,
                    Value::from(weight(m.name.as_deref().unwrap(), &m.aval.shape)),
                );
            }
            Storage::State | Storage::Device | Storage::Computed(_) => {}
        }
    }
    for &(si, _) in &g.state {
        inputs.insert(si, Value::from(HostTensor::zeros(g.aval(si).shape.clone())));
    }
    inputs
}

/// A larger model (12 layers, hidden 512, GQA 16:4, head_dim 64), so per-step timing is not dominated by
/// fixed graph-walk overhead.
fn model_big() -> MappedModel {
    Dense::new(Family::Qwen2)
        .vocab(4096)
        .dims(512, 1024, 12)
        .heads(16, 4)
        .head_dim(64)
        .max_positions(256)
        .f32_model()
}

/// Prefill host-path benches (the TTFT side; decode is below). Mirrors the decode benches (trace, cse+fuse,
/// one eager forward un-fused vs fused) over the whole-prompt prefill graph (a prefill step, `n` tokens
/// at once), whose cost scales with prompt length (n^2 attention + per-token FFN). `trace` and `cse+fuse` run
/// on the larger config (cheap graph-build passes); the eager `eval` pairs run on the small config at n=16
/// and n=64, since a `cfg_big` prefill eval takes seconds per iteration, too slow and noisy for a
/// microbench, while the small config stays in milliseconds and still shows length scaling and the fusion win.
fn prefill_benches(c: &mut Criterion) {
    let model_b = model_big();
    c.bench_function("trace_prefill", |b| {
        b.iter(|| {
            trace(
                black_box(&model_b),
                Phase::Prefill,
                black_box(64),
                black_box(64),
            )
        })
    });
    let gb = plain(trace(&model_b, Phase::Prefill, 64, 64));
    c.bench_function("prefill_cse+fuse", |b| {
        b.iter(|| transform::fuse(&transform::cse(black_box(&gb))))
    });

    // Eager prefill forward, un-fused vs fused, on the small config at two prompt lengths; each pair is the
    // prefill analogue of `eval_decode_step` / `_fused`.
    let model = model();
    for n in [16usize, 64usize] {
        let g = plain(trace(&model, Phase::Prefill, n, n));
        let inputs = bind(&g);
        let gf = transform::fuse(&transform::cse(&g));
        c.bench_function(&format!("eval_prefill_{n}"), |b| {
            b.iter(|| {
                poot_eval::eval(
                    black_box(&g),
                    black_box(&inputs),
                    EvalOptions::new(EvalBudget::UNBOUNDED),
                )
                .and_then(|evaluation| {
                    let state = evaluation
                        .state
                        .into_iter()
                        .map(Value::into_host)
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok((evaluation.output.into_host()?, state))
                })
                .unwrap()
            })
        });
        c.bench_function(&format!("eval_prefill_{n}_fused"), |b| {
            b.iter(|| {
                poot_eval::eval(
                    black_box(&gf),
                    black_box(&inputs),
                    EvalOptions::new(EvalBudget::UNBOUNDED),
                )
                .and_then(|evaluation| {
                    let state = evaluation
                        .state
                        .into_iter()
                        .map(Value::into_host)
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok((evaluation.output.into_host()?, state))
                })
                .unwrap()
            })
        });
    }
}

fn decode_benches(c: &mut Criterion) {
    let model = model();
    let cap = 16;

    c.bench_function("trace_decode_masked", |b| {
        b.iter(|| trace(black_box(&model), Phase::Decode, 1, black_box(cap)))
    });

    let g = plain(trace(&model, Phase::Decode, 1, cap));
    c.bench_function("cse+fuse", |b| {
        b.iter(|| transform::fuse(&transform::cse(black_box(&g))))
    });

    // Un-fused vs fused eager decode step on the same graph and inputs: fusion collapses the
    // pointwise/rmsnorm chains into Fused/FusedRow eqns, so eval walks fewer ops. A local proxy for the G5
    // fusion win (the launch-overhead win is measured on GPU/PTX).
    let inputs = bind(&g);
    c.bench_function("eval_decode_step", |b| {
        b.iter(|| {
            poot_eval::eval(
                black_box(&g),
                black_box(&inputs),
                EvalOptions::new(EvalBudget::UNBOUNDED),
            )
            .and_then(|evaluation| {
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })
            .unwrap()
        })
    });
    let gf = transform::fuse(&transform::cse(&g));
    c.bench_function("eval_decode_step_fused", |b| {
        b.iter(|| {
            poot_eval::eval(
                black_box(&gf),
                black_box(&inputs),
                EvalOptions::new(EvalBudget::UNBOUNDED),
            )
            .and_then(|evaluation| {
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })
            .unwrap()
        })
    });

    // Same pair at the larger config.
    let model_b = model_big();
    let cap_b = 64;
    let gb = plain(trace(&model_b, Phase::Decode, 1, cap_b));
    let inputs_b = bind(&gb);
    let gbf = transform::fuse(&transform::cse(&gb));
    c.bench_function("eval_decode_step_big", |b| {
        b.iter(|| {
            poot_eval::eval(
                black_box(&gb),
                black_box(&inputs_b),
                EvalOptions::new(EvalBudget::UNBOUNDED),
            )
            .and_then(|evaluation| {
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })
            .unwrap()
        })
    });
    c.bench_function("eval_decode_step_big_fused", |b| {
        b.iter(|| {
            poot_eval::eval(
                black_box(&gbf),
                black_box(&inputs_b),
                EvalOptions::new(EvalBudget::UNBOUNDED),
            )
            .and_then(|evaluation| {
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })
            .unwrap()
        })
    });
}

criterion_group!(benches, decode_benches, prefill_benches);
criterion_main!(benches);
