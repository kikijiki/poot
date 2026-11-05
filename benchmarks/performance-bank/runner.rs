//! The performance bank's measurement, shared by the backend examples (`performance_bank.rs` in poot-wgpu-device,
//! poot-rocm-device, poot-vulkan-device and poot-ptx-device include this file by path; each supplies only its device).
//!
//! One invocation measures one case of `cases.json` through the executor contract: it builds the case's
//! graph and deterministic inputs, loads and compiles it for the device's target, adds one entry per weight
//! copy, steps them in rotation (so a cold-weights case never finds its weights in cache), and writes the
//! raw per-repeat times with the realized plan, the observed workload, memory counters and the comparison
//! against an independent f64 reference. It decides no speed verdict and fills no identity: the bank's
//! harness (`benchmarks/harness/performance_bank.py record`) completes the receipt from the case definition
//! and refuses it when the observed workload differs.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use poot_executor::{Device, DeviceTime, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::test_support::const_weights::ConstWeights;
use poot_graph_ir::{Builder, FragmentDtype, Graph, MatmulPrecision, TensorType, ops};
use poot_graph_plan::staged::{
    DeviceId, DevicePlacement, ExpertPlacement, Partition, StagedProgram, TargetSet, compile_staged,
};
use poot_graph_plan::{CompileOptions, Target};
use poot_load::packed_safetensors::sha256_digest;
use poot_quant::StoredBytes;
use poot_quant::scalar;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::{HostTensor, HostView};
use serde_json::{Value, json};

/// Largest `|got - want|` allowed against the f64 reference, relative to `1 + |want|`: f32 accumulation
/// over a few thousand terms, reassociated differently by each backend's schedule.
const REFERENCE_TOLERANCE: f64 = 2e-3;

/// The same bound for a case whose operands are narrowed: a matrix unit sums its products in its own order and
/// width (the measured residual against the rounding reference is a few times the f32 schedules'), so the
/// elementwise bound near zero is the narrow-storage class's (ADR-0101), not the f32 schedules'.
const NARROW_REFERENCE_TOLERANCE: f64 = 5e-3;

struct Args {
    bank: PathBuf,
    case: String,
    out: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let (mut bank, mut case, mut out) = (None, None, None);
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--bank" => bank = Some(PathBuf::from(value)),
            "--case" => case = Some(value),
            "--out" => out = Some(PathBuf::from(value)),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        bank: bank.ok_or("--bank is required")?,
        case: case.ok_or("--case is required")?,
        out: out.ok_or("--out is required")?,
    })
}

/// `splitmix64`, the bank's one token-fixture generator: uniform f32 in [-1, 1) from the top 24 bits.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    fn tensor(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next_unit()).collect()
    }
}

fn dim(shape: &Value, name: &str) -> Result<usize, String> {
    shape[name]
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| format!("case shape has no integer {name:?}"))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The operand rounding a case declares in `numerics.operand_precision` (`"exact"`, or `"narrow"`: operands enter
/// the products rounded to the fragment dtype of the backend's matrix unit, accumulated in f32). The dtype is the
/// backend's: bf16 where the unit takes it (ROCm, PTX), f16 on the Vulkan cooperative matrix (wgpu, raw Vulkan). A narrow case
/// compares with a reference that rounds the same operands, so its tolerance is the narrow-storage one ([`NARROW_REFERENCE_TOLERANCE`]).
#[derive(Clone, Copy)]
struct Operands {
    narrow: Option<FragmentDtype>,
}

impl Operands {
    fn of(backend: &str, case: &Value) -> Result<Self, String> {
        let narrow = match case["numerics"]["operand_precision"].as_str() {
            None | Some("exact") => None,
            Some("narrow") => Some(if matches!(backend, "wgpu" | "vulkan") {
                FragmentDtype::F16
            } else {
                FragmentDtype::Bf16
            }),
            Some(other) => return Err(format!("unknown operand_precision {other:?}")),
        };
        Ok(Self { narrow })
    }

    fn tolerance(self) -> f64 {
        match self.narrow {
            Some(_) => NARROW_REFERENCE_TOLERANCE,
            None => REFERENCE_TOLERANCE,
        }
    }

    fn policy(self) -> MatmulPrecision {
        match self.narrow {
            Some(_) => MatmulPrecision::Narrow,
            None => MatmulPrecision::Exact,
        }
    }

    /// `v` as the products read it: rounded to nearest even to the fragment dtype, written here from the format
    /// definitions and sharing no code with the engine.
    fn round(self, v: f32) -> f64 {
        match self.narrow {
            None => f64::from(v),
            Some(FragmentDtype::Bf16) => {
                let bits = v.to_bits();
                let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
                f64::from(f32::from_bits(rounded))
            }
            Some(FragmentDtype::F16) => {
                // The inputs are in [-1, 1): a binary16 has a 2^-24 grid below 2^-14 and 11 significant bits
                // above it.
                let magnitude = f64::from(v.abs());
                let exponent = magnitude.max(2f64.powi(-14)).log2().floor();
                let step = 2f64.powf(exponent - 10.0);
                (magnitude / step).round_ties_even() * step * f64::from(v.signum())
            }
        }
    }
}

/// One case ready to run: graphs (one per weight copy), their consts, and the reference output per copy.
struct Prepared {
    graphs: Vec<Graph>,
    consts: Vec<(String, HostTensor)>,
    references: Vec<Vec<f64>>,
    out_shape: Vec<usize>,
    /// Per-step slot inputs, bound identically at every step: key, shape and the I32 words as bytes.
    slots: Vec<(poot_graph_ir::SlotKey, Vec<usize>, Vec<u8>)>,
    observed: Value,
}

fn weight_copies(bank: &Value, case: &Value, weight_bytes: usize) -> Result<usize, String> {
    let condition = &bank["cache_conditions"][case["cache"].as_str().ok_or("case has no cache")?];
    match condition["kind"].as_str() {
        Some("weight-rotation") => {
            let min = condition["min_rotation_bytes"]
                .as_u64()
                .ok_or("rotation condition has no min_rotation_bytes")?
                as usize;
            Ok(min.div_ceil(weight_bytes))
        }
        Some("single-copy") => Ok(1),
        other => Err(format!("unknown cache condition kind {other:?}")),
    }
}

fn dense_matmul(bank: &Value, case: &Value, ops: Operands) -> Result<Prepared, String> {
    let shape = &case["shape"];
    let (m, k, n) = (dim(shape, "m")?, dim(shape, "k")?, dim(shape, "n")?);
    let copies = weight_copies(bank, case, k * n * 4)?;
    let mut stream = SplitMix64(
        bank["token_fixtures"][case["token_fixture"].as_str().unwrap()]["seed"]
            .as_u64()
            .ok_or("fixture has no seed")?,
    );
    let x = stream.tensor(m * k);
    let x_read: Vec<f64> = x.iter().map(|&v| ops.round(v)).collect();
    let mut consts = vec![("x".to_string(), HostTensor::f32(vec![m, k], x.clone()))];
    let (mut graphs, mut references) = (Vec::new(), Vec::new());
    for copy in 0..copies {
        let w = stream.tensor(k * n);
        let mut want = vec![0.0f64; m * n];
        for row in 0..m {
            for kk in 0..k {
                let a = x_read[row * k + kk];
                for (col, acc) in want[row * n..(row + 1) * n].iter_mut().enumerate() {
                    *acc += a * ops.round(w[kk * n + col]);
                }
            }
        }
        references.push(want);
        let name = format!("w{copy}");
        let b = Builder::new(poot_graph_ir::TraceLimits::for_tests());
        let xi = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
        let wi = b.constant(&name, TensorType::f32(vec![k, n])).unwrap();
        let y = b.matmul(xi, wi).unwrap();
        graphs.push(b.finish(y).unwrap());
        consts.push((name, HostTensor::f32(vec![k, n], w)));
    }
    Ok(Prepared {
        graphs,
        consts,
        references,
        out_shape: vec![m, n],
        slots: Vec::new(),
        observed: json!({
            "shape": { "m": m, "k": k, "n": n },
            "storage": "f32",
            "layout": "kn",
            "weight_copies": copies,
            "weight_bytes_per_copy": k * n * 4,
        }),
    })
}

fn attention_decode(bank: &Value, case: &Value) -> Result<Prepared, String> {
    let shape = &case["shape"];
    let (hq, hkv, d, cap) = (
        dim(shape, "hq")?,
        dim(shape, "hkv")?,
        dim(shape, "d")?,
        dim(shape, "cap")?,
    );
    // `live`, when the shape names it, is the number of keys the step attends to: the graph derives its
    // length from `Slot::Pos` (the guarded live composite), so a short live length against a long capacity
    // is what a decode step at that position runs. Without it the whole capacity is live.
    let live = match shape.get("live") {
        None => None,
        Some(value) => Some(
            value
                .as_u64()
                .map(|v| v as usize)
                .filter(|&v| (1..=cap).contains(&v))
                .ok_or("case shape `live` must be an integer in 1..=cap")?,
        ),
    };
    let attended = live.unwrap_or(cap);
    let copies = weight_copies(bank, case, 2 * hkv * cap * d * 4)?;
    if copies != 1 {
        return Err(
            "attention_decode holds its KV resident: use a single-copy cache condition".into(),
        );
    }
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let mut stream = SplitMix64(
        bank["token_fixtures"][case["token_fixture"].as_str().unwrap()]["seed"]
            .as_u64()
            .ok_or("fixture has no seed")?,
    );
    let q = stream.tensor(hq * d);
    let k = stream.tensor(hkv * cap * d);
    let v = stream.tensor(hkv * cap * d);
    let mut want = vec![0.0f64; hq * d];
    for head in 0..hq {
        let kv = head / n_rep;
        let scores: Vec<f64> = (0..attended)
            .map(|t| {
                let dot: f64 = (0..d)
                    .map(|i| f64::from(q[head * d + i]) * f64::from(k[(kv * cap + t) * d + i]))
                    .sum();
                dot * f64::from(scale)
            })
            .collect();
        let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - peak).exp()).collect();
        let total: f64 = weights.iter().sum();
        for i in 0..d {
            want[head * d + i] = (0..attended)
                .map(|t| weights[t] * f64::from(v[(kv * cap + t) * d + i]))
                .sum::<f64>()
                / total;
        }
    }
    let b = Builder::new(poot_graph_ir::TraceLimits::for_tests());
    let qi = b.constant("q", TensorType::f32(vec![1, hq, 1, d])).unwrap();
    let ki = b
        .constant("k", TensorType::f32(vec![1, hkv, cap, d]))
        .unwrap();
    let vi = b
        .constant("v", TensorType::f32(vec![1, hkv, cap, d]))
        .unwrap();
    let mut consts = vec![
        ("q".into(), HostTensor::f32(vec![1, hq, 1, d], q)),
        ("k".into(), HostTensor::f32(vec![1, hkv, cap, d], k)),
        ("v".into(), HostTensor::f32(vec![1, hkv, cap, d], v)),
    ];
    let mut slots = Vec::new();
    let graph = if live.is_some() {
        let pos = b
            .slot_ranged(
                poot_graph_ir::Slot::Pos,
                TensorType::new(vec![1, 1], poot_tensor::DType::I32),
                // The position is declared in `0..cap`, which every binder checks: the guarded composite then
                // needs no validation witness to fuse to the live attention (the serving decode graphs' form).
                poot_graph_ir::IndexRange::below(cap).ok_or("cap must be non-zero")?,
            )
            .unwrap();
        let mask = ops::causal_mask_from_pos(&b, pos, cap, None).unwrap();
        let len = b
            .binary_scalar(
                poot_graph_ir::BinOp::Add,
                b.reshape(pos, vec![1]).unwrap(),
                poot_graph_ir::Scalar::I32(1),
            )
            .unwrap();
        let composite = ops::attention_live(
            &b,
            qi,
            ops::AttentionKeys::Single(ki),
            vi,
            n_rep,
            scale,
            mask,
            None,
            len,
        )
        .map_err(|e| format!("the live attention composite: {e:?}"))?;
        let graph = b.finish(composite.out).unwrap();
        let key = graph
            .meta(pos.id())
            .slot_key()
            .cloned()
            .ok_or("Slot::Pos has no slot key")?;
        slots.push((
            key,
            vec![1, 1],
            ((attended - 1) as i32).to_le_bytes().to_vec(),
        ));
        graph
    } else {
        let mask = b
            .constant("mask", TensorType::f32(vec![1, 1, 1, cap]))
            .unwrap();
        consts.push((
            "mask".into(),
            HostTensor::f32(vec![1, 1, 1, cap], vec![0.0; cap]),
        ));
        let y = ops::attention_masked(&b, qi, ki, vi, n_rep, scale, mask).unwrap();
        b.finish(y).unwrap()
    };
    let mut shape_json = json!({ "hq": hq, "hkv": hkv, "d": d, "cap": cap });
    if let Some(live) = live {
        shape_json["live"] = json!(live);
    }
    Ok(Prepared {
        graphs: vec![graph],
        consts,
        references: vec![want],
        out_shape: vec![1, hq, 1, d],
        slots,
        observed: json!({
            "shape": shape_json,
            "storage": "f32",
            "layout": "bhsd",
            "weight_copies": 1,
            "weight_bytes_per_copy": 2 * hkv * cap * d * 4,
        }),
    })
}

/// A dense contraction over a checkpoint-order `[N, K]` BF16 weight, in the form a model's `linear` traces
/// (`MatMul(x, Transpose(w))`, which compile folds into one `DenseContraction` reading `w` as stored). The
/// f32 activation `x` is `[M, K]`. Each weight is the fixture's f32 draw truncated to its top 16 bits, so the
/// stored BF16 value is exact and the reference reads it back by the same bit shift, with no engine code.
fn dense_contraction(bank: &Value, case: &Value, ops: Operands) -> Result<Prepared, String> {
    let shape = &case["shape"];
    let (m, k, n) = (dim(shape, "m")?, dim(shape, "k")?, dim(shape, "n")?);
    if case["storage"] != "bf16" || case["layout"] != "nk" {
        return Err("dense_contraction measures a bf16 [N, K] weight".into());
    }
    let copies = weight_copies(bank, case, n * k * 2)?;
    let mut stream = SplitMix64(
        bank["token_fixtures"][case["token_fixture"].as_str().unwrap()]["seed"]
            .as_u64()
            .ok_or("fixture has no seed")?,
    );
    let x = stream.tensor(m * k);
    let x_read: Vec<f64> = x.iter().map(|&v| ops.round(v)).collect();
    let mut consts = vec![("x".to_string(), HostTensor::f32(vec![m, k], x.clone()))];
    let (mut graphs, mut references) = (Vec::new(), Vec::new());
    for copy in 0..copies {
        let words: Vec<u16> = stream
            .tensor(n * k)
            .into_iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let w: Vec<f64> = words
            .iter()
            .map(|&bits| ops.round(f32::from_bits(u32::from(bits) << 16)))
            .collect();
        let mut want = vec![0.0f64; m * n];
        for row in 0..m {
            let xs = &x_read[row * k..(row + 1) * k];
            for (col, acc) in want[row * n..(row + 1) * n].iter_mut().enumerate() {
                *acc = xs
                    .iter()
                    .zip(&w[col * k..(col + 1) * k])
                    .map(|(&a, &b)| a * b)
                    .sum();
            }
        }
        references.push(want);
        let name = format!("w{copy}");
        let b = Builder::new(poot_graph_ir::TraceLimits::for_tests());
        let xi = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
        let wi = b
            .constant(&name, TensorType::new(vec![n, k], poot_tensor::DType::BF16))
            .unwrap();
        let y = b.matmul(xi, b.transpose(wi, vec![1, 0]).unwrap()).unwrap();
        graphs.push(b.finish(y).unwrap());
        consts.push((name, HostTensor::bf16(vec![n, k], words)));
    }
    Ok(Prepared {
        graphs,
        consts,
        references,
        out_shape: vec![m, n],
        slots: Vec::new(),
        observed: json!({
            "shape": { "m": m, "k": k, "n": n },
            "storage": "bf16",
            "layout": "nk",
            "weight_copies": copies,
            "weight_bytes_per_copy": n * k * 2,
        }),
    })
}

/// One routed expert projection of a MoE prefill: `tokens * top_k` rows (each a token at one of its slots) through
/// the indexed contraction against a stack of `experts` weights, the rows in row order as a model traces them
/// (`compile` groups them by expert for a tiled contraction). Each row's expert is drawn from the fixture stream,
/// uniform over the experts (`order: "routed"` or `"shuffled"`), or the same draws sorted ascending (`"sorted"`).
/// The stack is f32 `[E, K, N]` (`kn`) or the checkpoint's bf16 `[E, N, K]` (`nk`, read through its transposed
/// view). The f64 reference contracts each row against its own expert.
fn grouped_experts(bank: &Value, case: &Value, ops: Operands) -> Result<Prepared, String> {
    let shape = &case["shape"];
    let (tokens, experts, k, n) = (
        dim(shape, "tokens")?,
        dim(shape, "experts")?,
        dim(shape, "k")?,
        dim(shape, "n")?,
    );
    let top_k = shape["top_k"].as_u64().map_or(1, |v| v as usize);
    let rows = tokens * top_k;
    let nk = match (case["storage"].as_str(), case["layout"].as_str()) {
        (Some("f32"), Some("kn")) => false,
        (Some("bf16"), Some("nk")) => true,
        other => {
            return Err(format!(
                "grouped_experts measures f32 kn or bf16 nk, got {other:?}"
            ));
        }
    };
    let elem_bytes = if nk { 2 } else { 4 };
    let copies = weight_copies(bank, case, experts * k * n * elem_bytes)?;
    let mut stream = SplitMix64(
        bank["token_fixtures"][case["token_fixture"].as_str().unwrap()]["seed"]
            .as_u64()
            .ok_or("fixture has no seed")?,
    );
    // One activation row per token; routed row `r` is token `r / top_k` at its slot `r % top_k`.
    let x = stream.tensor(tokens * k);
    let x_read: Vec<f64> = x.iter().map(|&v| ops.round(v)).collect();
    let mut ids: Vec<i32> = stream
        .tensor(rows)
        .into_iter()
        .map(|u| ((((u + 1.0) * 0.5) * experts as f32) as usize).min(experts - 1) as i32)
        .collect();
    match shape["order"].as_str() {
        Some("routed" | "shuffled") => {}
        Some("sorted") => ids.sort_unstable(),
        other => return Err(format!("unknown routing order {other:?}")),
    }
    let mut consts = vec![("x".to_string(), HostTensor::f32(vec![tokens, k], x.clone()))];
    let (mut graphs, mut references, mut slots) = (Vec::new(), Vec::new(), Vec::new());
    for copy in 0..copies {
        let draws = stream.tensor(experts * k * n);
        let name = format!("w{copy}");
        // `w_read(e, kk, col)`: the weight value the products read, from the stored layout.
        let (stored, w_read): (HostTensor, Vec<f64>) = if nk {
            let words: Vec<u16> = draws.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let read = words
                .iter()
                .map(|&bits| ops.round(f32::from_bits(u32::from(bits) << 16)))
                .collect();
            (HostTensor::bf16(vec![experts, n, k], words), read)
        } else {
            let read = draws.iter().map(|&v| ops.round(v)).collect();
            (HostTensor::f32(vec![experts, k, n], draws), read)
        };
        let at = |e: usize, kk: usize, col: usize| {
            if nk {
                (e * n + col) * k + kk
            } else {
                (e * k + kk) * n + col
            }
        };
        let mut want = vec![0.0f64; rows * n];
        for row in 0..rows {
            let (e, token) = (ids[row] as usize, row / top_k);
            let xs = &x_read[token * k..(token + 1) * k];
            for (col, acc) in want[row * n..(row + 1) * n].iter_mut().enumerate() {
                *acc = xs
                    .iter()
                    .enumerate()
                    .map(|(kk, &a)| a * w_read[at(e, kk, col)])
                    .sum();
            }
        }
        references.push(want);
        let b = Builder::new(poot_graph_ir::TraceLimits::for_tests());
        // The routed rows as the MoE block traces them: each token's row broadcast over its `top_k` slots.
        let tokens_x = b.constant("x", TensorType::f32(vec![tokens, k])).unwrap();
        let xi = b
            .reshape(
                b.broadcast(
                    b.reshape(tokens_x, vec![tokens, 1, k]).unwrap(),
                    vec![tokens, top_k, k],
                )
                .unwrap(),
                vec![rows, k],
            )
            .unwrap();
        let wi = if nk {
            let stack = b
                .constant(
                    &name,
                    TensorType::new(vec![experts, n, k], poot_tensor::DType::BF16),
                )
                .unwrap();
            b.transpose(stack, vec![0, 2, 1]).unwrap()
        } else {
            b.constant(&name, TensorType::f32(vec![experts, k, n]))
                .unwrap()
        };
        let idi = b
            .slot_named_ranged(
                poot_graph_ir::Slot::Activation,
                "ids",
                TensorType::new(vec![rows], poot_tensor::DType::I32),
                poot_graph_ir::IndexRange::below(experts).ok_or("experts must be non-zero")?,
            )
            .unwrap();
        let y = b.indexed_matmul(xi, wi, idi).unwrap();
        let graph = b.finish(y).unwrap();
        // Every copy's graph declares the one ids slot alike; it is bound once per step.
        if slots.is_empty() {
            let key = graph
                .meta(idi.id())
                .slot_key()
                .cloned()
                .ok_or("the ids slot has no slot key")?;
            let bytes = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            slots.push((key, vec![rows], bytes));
        }
        graphs.push(graph);
        consts.push((name, stored));
    }
    Ok(Prepared {
        graphs,
        consts,
        references,
        out_shape: vec![rows, n],
        slots,
        observed: json!({
            "shape": shape.clone(),
            "storage": case["storage"].clone(),
            "layout": case["layout"].clone(),
            "weight_copies": copies,
            "weight_bytes_per_copy": experts * k * n * elem_bytes,
        }),
    })
}

/// Causal prefill attention: `lq` query rows at the last `lq` positions of a `cap`-key context, under the
/// causal mask a model derives from `Slot::Pos` (`causal_mask_from_pos`), so the planner sees the prefill
/// graph a model traces. GQA with `hq / hkv` query heads per KV head. The f64 reference attends each query at
/// position `p` to keys `0..=p`, one thread per head.
fn attention_prefill(bank: &Value, case: &Value) -> Result<Prepared, String> {
    let shape = &case["shape"];
    let (hq, hkv, d, lq, cap) = (
        dim(shape, "hq")?,
        dim(shape, "hkv")?,
        dim(shape, "d")?,
        dim(shape, "lq")?,
        dim(shape, "cap")?,
    );
    if !(1..=cap).contains(&lq) || hkv == 0 || hq % hkv != 0 {
        return Err("attention_prefill needs 1 <= lq <= cap and hkv dividing hq".into());
    }
    // The case's `storage` is the key/value cache's dtype: f32, or the engine's default f16.
    let kv_f16 = match case["storage"].as_str() {
        Some("f32") => false,
        Some("f16") => true,
        other => {
            return Err(format!(
                "attention_prefill stores its KV as f32 or f16, not {other:?}"
            ));
        }
    };
    let kv_bytes = if kv_f16 { 2 } else { 4 };
    let copies = weight_copies(bank, case, 2 * hkv * cap * d * kv_bytes)?;
    if copies != 1 {
        return Err(
            "attention_prefill holds its KV resident: use a single-copy cache condition".into(),
        );
    }
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let first = cap - lq;
    let mut stream = SplitMix64(
        bank["token_fixtures"][case["token_fixture"].as_str().unwrap()]["seed"]
            .as_u64()
            .ok_or("fixture has no seed")?,
    );
    let q = stream.tensor(hq * lq * d);
    let mut k = stream.tensor(hkv * cap * d);
    let mut v = stream.tensor(hkv * cap * d);
    // An f16 cache holds the rounded values: the reference reads exactly what the device reads.
    let k_words: Vec<u16> = if kv_f16 {
        k.iter().map(|&x| scalar::f32_to_f16(x)).collect()
    } else {
        Vec::new()
    };
    let v_words: Vec<u16> = if kv_f16 {
        v.iter().map(|&x| scalar::f32_to_f16(x)).collect()
    } else {
        Vec::new()
    };
    if kv_f16 {
        k = k_words.iter().map(|&w| scalar::f16_to_f32(w)).collect();
        v = v_words.iter().map(|&w| scalar::f16_to_f32(w)).collect();
    }
    let mut want = vec![0.0f64; hq * lq * d];
    std::thread::scope(|scope| {
        for (head, out) in want.chunks_mut(lq * d).enumerate() {
            let (q, k, v) = (&q, &k, &v);
            scope.spawn(move || {
                let kv = head / n_rep;
                let mut scores = vec![0.0f64; cap];
                for row in 0..lq {
                    let visible = first + row + 1;
                    let qr = &q[(head * lq + row) * d..(head * lq + row + 1) * d];
                    for (t, score) in scores[..visible].iter_mut().enumerate() {
                        let kr = &k[(kv * cap + t) * d..(kv * cap + t + 1) * d];
                        let dot: f64 = qr
                            .iter()
                            .zip(kr)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum();
                        *score = dot * f64::from(scale);
                    }
                    let peak = scores[..visible]
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let mut total = 0.0;
                    let o = &mut out[row * d..(row + 1) * d];
                    for (t, score) in scores[..visible].iter().enumerate() {
                        let weight = (score - peak).exp();
                        total += weight;
                        let vr = &v[(kv * cap + t) * d..(kv * cap + t + 1) * d];
                        for (acc, &value) in o.iter_mut().zip(vr) {
                            *acc += weight * f64::from(value);
                        }
                    }
                    o.iter_mut().for_each(|acc| *acc /= total);
                }
            });
        }
    });
    let b = Builder::new(poot_graph_ir::TraceLimits::for_tests());
    let qi = b
        .constant("q", TensorType::f32(vec![1, hq, lq, d]))
        .unwrap();
    let kv_type = |shape: Vec<usize>| {
        if kv_f16 {
            TensorType::f16(shape)
        } else {
            TensorType::f32(shape)
        }
    };
    let ki = b.constant("k", kv_type(vec![1, hkv, cap, d])).unwrap();
    let vi = b.constant("v", kv_type(vec![1, hkv, cap, d])).unwrap();
    let pos = b
        .slot_ranged(
            poot_graph_ir::Slot::Pos,
            TensorType::new(vec![1, lq], poot_tensor::DType::I32),
            poot_graph_ir::IndexRange::below(cap).ok_or("cap must be non-zero")?,
        )
        .unwrap();
    let mask = ops::causal_mask_from_pos(&b, pos, cap, None).unwrap();
    let y = ops::attention_masked(&b, qi, ki, vi, n_rep, scale, mask).unwrap();
    let graph = b.finish(y).unwrap();
    let key = graph
        .meta(pos.id())
        .slot_key()
        .cloned()
        .ok_or("Slot::Pos has no slot key")?;
    let positions: Vec<u8> = (first..cap)
        .flat_map(|p| (p as i32).to_le_bytes())
        .collect();
    Ok(Prepared {
        graphs: vec![graph],
        consts: vec![
            ("q".into(), HostTensor::f32(vec![1, hq, lq, d], q)),
            (
                "k".into(),
                if kv_f16 {
                    HostTensor::f16(vec![1, hkv, cap, d], k_words)
                } else {
                    HostTensor::f32(vec![1, hkv, cap, d], k)
                },
            ),
            (
                "v".into(),
                if kv_f16 {
                    HostTensor::f16(vec![1, hkv, cap, d], v_words)
                } else {
                    HostTensor::f32(vec![1, hkv, cap, d], v)
                },
            ),
        ],
        references: vec![want],
        out_shape: vec![1, hq, lq, d],
        slots: vec![(key, vec![1, lq], positions)],
        observed: json!({
            "shape": { "hq": hq, "hkv": hkv, "d": d, "lq": lq, "cap": cap },
            "storage": if kv_f16 { "f16" } else { "f32" },
            "layout": "bhsd",
            "weight_copies": 1,
            "weight_bytes_per_copy": 2 * hkv * cap * d * kv_bytes,
        }),
    })
}

/// The recurrence scan of one family. A case's shape names `"family": "ssd"` for the Mamba2 SSD (`p` head
/// width, `n` state rows); a shape without a family is the gated delta rule (`d`).
fn recurrence_scan(bank: &Value, case: &Value) -> Result<Prepared, String> {
    match case["shape"]["family"].as_str() {
        None => delta_rule_scan(bank, case),
        Some("ssd") => ssd_scan(bank, case),
        Some(other) => Err(format!("unknown scan family {other:?}")),
    }
}

/// The Mamba2 SSD over `steps` tokens (`heads` heads, head width `p`, state `n x p`). `delta` is a small
/// positive step, `a` a negative per-head decay rate, so the state stays bounded; the reference is a plain
/// f64 loop of the kernel's per-column recurrence. The graph's output is the packed `[1, H, L + N, P]` result.
fn ssd_scan(bank: &Value, case: &Value) -> Result<Prepared, String> {
    let shape = &case["shape"];
    let (heads, p, n, len) = (
        dim(shape, "heads")?,
        dim(shape, "p")?,
        dim(shape, "n")?,
        dim(shape, "steps")?,
    );
    let mut stream = SplitMix64(
        bank["token_fixtures"][case["token_fixture"].as_str().unwrap()]["seed"]
            .as_u64()
            .ok_or("fixture has no seed")?,
    );
    let x = stream.tensor(heads * len * p);
    let b_in: Vec<f32> = stream
        .tensor(heads * len * n)
        .into_iter()
        .map(|u| u * 0.5)
        .collect();
    let c_in: Vec<f32> = stream
        .tensor(heads * len * n)
        .into_iter()
        .map(|u| u * 0.5)
        .collect();
    // delta in [0.05, 0.15), a in [-0.7, -0.4) per head, d in [0.5, 0.8).
    let delta: Vec<f32> = stream
        .tensor(heads * len)
        .into_iter()
        .map(|u| 0.1 + 0.05 * u)
        .collect();
    let a: Vec<f32> = (0..heads)
        .map(|h| -(0.4 + 0.3 * h as f32 / heads as f32))
        .collect();
    let d_skip: Vec<f32> = (0..heads)
        .map(|h| 0.5 + 0.3 * h as f32 / heads as f32)
        .collect();
    let s_in = vec![0.0f32; heads * n * p];
    let rows = len + n;
    let mut want = vec![0.0f64; heads * rows * p];
    for h in 0..heads {
        for col in 0..p {
            let mut s = vec![0.0f64; n];
            for t in 0..len {
                let decay = (f64::from(delta[h * len + t]) * f64::from(a[h])).exp();
                let dx = f64::from(delta[h * len + t]) * f64::from(x[(h * len + t) * p + col]);
                let bc = (h * len + t) * n;
                let mut y = 0.0;
                for r in 0..n {
                    s[r] = decay * s[r] + f64::from(b_in[bc + r]) * dx;
                    y += f64::from(c_in[bc + r]) * s[r];
                }
                y += f64::from(d_skip[h]) * f64::from(x[(h * len + t) * p + col]);
                want[(h * rows + t) * p + col] = y;
            }
            for r in 0..n {
                want[(h * rows + len + r) * p + col] = s[r];
            }
        }
    }
    let b = Builder::new(poot_graph_ir::TraceLimits::for_tests());
    let x_ty = vec![1, heads, len, p];
    let bc_ty = vec![1, heads, len, n];
    let delta_ty = vec![1, heads, len, 1];
    let head_ty = vec![1, heads, 1, 1];
    let state_ty = vec![1, heads, n, p];
    let operands = [
        b.constant("x", TensorType::f32(x_ty.clone())).unwrap(),
        b.constant("b", TensorType::f32(bc_ty.clone())).unwrap(),
        b.constant("c", TensorType::f32(bc_ty.clone())).unwrap(),
        b.constant("delta", TensorType::f32(delta_ty.clone()))
            .unwrap(),
        b.constant("a", TensorType::f32(head_ty.clone())).unwrap(),
        b.constant("d", TensorType::f32(head_ty.clone())).unwrap(),
        b.constant("s_in", TensorType::f32(state_ty.clone()))
            .unwrap(),
    ];
    let scan = b
        .scan(
            poot_graph_ir::ScanAttrs {
                family: poot_graph_ir::ScanFamily::Ssd,
                chunk: 64,
            },
            &operands,
        )
        .unwrap();
    Ok(Prepared {
        graphs: vec![b.finish(scan.packed).unwrap()],
        consts: vec![
            ("x".into(), HostTensor::f32(x_ty, x)),
            ("b".into(), HostTensor::f32(bc_ty.clone(), b_in)),
            ("c".into(), HostTensor::f32(bc_ty, c_in)),
            ("delta".into(), HostTensor::f32(delta_ty, delta)),
            ("a".into(), HostTensor::f32(head_ty.clone(), a)),
            ("d".into(), HostTensor::f32(head_ty, d_skip)),
            ("s_in".into(), HostTensor::f32(state_ty, s_in)),
        ],
        references: vec![want],
        out_shape: vec![1, heads, rows, p],
        slots: Vec::new(),
        observed: json!({
            "shape": { "family": "ssd", "heads": heads, "p": p, "n": n, "steps": len },
            "storage": "f32",
            "layout": "bhsd",
            "weight_copies": 1,
            "weight_bytes_per_copy": 4 * (heads * len * (p + 2 * n + 1) + 2 * heads + heads * n * p),
        }),
    })
}

/// The gated delta rule over `steps` tokens (`heads` heads, state `d x d`), the bank's recurrence. Keys are
/// normalised to unit length and `g`, `beta` kept in a contracting range so the 4096-step state stays bounded;
/// the reference is a plain f64 loop of the kernel's per-column recurrence, sharing no code with the engine.
/// The graph's output is the packed `[1, H, L + D, D]` result, so every output row and the carried state are
/// compared.
fn delta_rule_scan(bank: &Value, case: &Value) -> Result<Prepared, String> {
    let shape = &case["shape"];
    let (heads, d, len) = (dim(shape, "heads")?, dim(shape, "d")?, dim(shape, "steps")?);
    let mut stream = SplitMix64(
        bank["token_fixtures"][case["token_fixture"].as_str().unwrap()]["seed"]
            .as_u64()
            .ok_or("fixture has no seed")?,
    );
    let q = stream.tensor(heads * len * d);
    let mut k = stream.tensor(heads * len * d);
    let v = stream.tensor(heads * len * d);
    for row in k.chunks_mut(d) {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        row.iter_mut().for_each(|x| *x /= norm);
    }
    // g in [-0.30, -0.05) (log-domain decay), beta in [0.1, 0.9).
    let g: Vec<f32> = stream
        .tensor(heads * len)
        .into_iter()
        .map(|u| -0.175 + 0.125 * u)
        .collect();
    let beta: Vec<f32> = stream
        .tensor(heads * len)
        .into_iter()
        .map(|u| 0.5 + 0.4 * u)
        .collect();
    let s_in = vec![0.0f32; heads * d * d];
    let scale = 1.0 / (d as f64).sqrt();
    let rows = len + d;
    let mut want = vec![0.0f64; heads * rows * d];
    for h in 0..heads {
        for col in 0..d {
            let mut s = vec![0.0f64; d];
            for t in 0..len {
                let at = (h * len + t) * d;
                let decay = f64::from(g[h * len + t]).exp();
                let kv: f64 = (0..d).map(|r| f64::from(k[at + r]) * decay * s[r]).sum();
                let delta = f64::from(beta[h * len + t]) * (f64::from(v[at + col]) - kv);
                let mut o = 0.0;
                for r in 0..d {
                    s[r] = decay * s[r] + f64::from(k[at + r]) * delta;
                    o += f64::from(q[at + r]) * scale * s[r];
                }
                want[(h * rows + t) * d + col] = o;
            }
            for r in 0..d {
                want[(h * rows + len + r) * d + col] = s[r];
            }
        }
    }
    let b = Builder::new(poot_graph_ir::TraceLimits::for_tests());
    let sequence = vec![1, heads, len, d];
    let per_token = vec![1, heads, len, 1];
    let operands = [
        b.constant("q", TensorType::f32(sequence.clone())).unwrap(),
        b.constant("k", TensorType::f32(sequence.clone())).unwrap(),
        b.constant("v", TensorType::f32(sequence.clone())).unwrap(),
        b.constant("g", TensorType::f32(per_token.clone())).unwrap(),
        b.constant("beta", TensorType::f32(per_token.clone()))
            .unwrap(),
        b.constant("s_in", TensorType::f32(vec![1, heads, d, d]))
            .unwrap(),
    ];
    let scan = b
        .scan(
            poot_graph_ir::ScanAttrs {
                family: poot_graph_ir::ScanFamily::DeltaRule,
                chunk: 64,
            },
            &operands,
        )
        .unwrap();
    Ok(Prepared {
        graphs: vec![b.finish(scan.packed).unwrap()],
        consts: vec![
            ("q".into(), HostTensor::f32(sequence.clone(), q)),
            ("k".into(), HostTensor::f32(sequence.clone(), k)),
            ("v".into(), HostTensor::f32(sequence, v)),
            ("g".into(), HostTensor::f32(per_token.clone(), g)),
            ("beta".into(), HostTensor::f32(per_token, beta)),
            ("s_in".into(), HostTensor::f32(vec![1, heads, d, d], s_in)),
        ],
        references: vec![want],
        out_shape: vec![1, heads, rows, d],
        slots: Vec::new(),
        observed: json!({
            "shape": { "heads": heads, "d": d, "steps": len },
            "storage": "f32",
            "layout": "bhsd",
            "weight_copies": 1,
            "weight_bytes_per_copy": 4 * (3 * heads * len * d + 2 * heads * len + heads * d * d),
        }),
    })
}

fn prepare(bank: &Value, case: &Value, ops: Operands) -> Result<Prepared, String> {
    let generator = &bank["token_fixtures"][case["token_fixture"]
        .as_str()
        .ok_or("case has no token_fixture")?]["generator"];
    if generator != "splitmix64" {
        return Err(format!("unsupported token fixture generator {generator}"));
    }
    match case["producer"].as_str() {
        Some("dense_matmul") => dense_matmul(bank, case, ops),
        Some("dense_contraction") => dense_contraction(bank, case, ops),
        Some("attention_decode") => attention_decode(bank, case),
        Some("attention_prefill") => attention_prefill(bank, case),
        Some("recurrence_scan") => recurrence_scan(bank, case),
        Some("grouped_experts") => grouped_experts(bank, case, ops),
        other => Err(format!("no runner for producer {other:?}")),
    }
}

fn staged(g: &Graph, target: Target, policy: MatmulPrecision) -> Result<StagedProgram, String> {
    compile_staged(
        &g.clone(),
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD)
            .with_matmul_precision(policy),
    )
    .map_err(|e| format!("compile_staged: {e}"))
}

/// The kernels the planner chose, in plan order: label (family, schedule, shape) and plan key.
fn plan_of(program: &StagedProgram) -> Vec<String> {
    program
        .stages()
        .flat_map(|(_, _, p)| p.planned())
        .flat_map(|(_, plan)| {
            plan.dispatches()
                .iter()
                .map(|d| format!("{} [{}]", d.label(), d.key()))
        })
        .collect()
}

fn reference_check(got: &[f32], want: &[f64], tolerance: f64) -> (f64, bool) {
    let mut worst = 0.0f64;
    let mut agrees = got.len() == want.len();
    for (g, w) in got.iter().zip(want) {
        let err = (f64::from(*g) - w).abs() / (1.0 + w.abs());
        // NaN never agrees (ADR-0101).
        if err.is_nan() || err > tolerance {
            agrees = false;
        }
        worst = worst.max(if err.is_nan() { f64::INFINITY } else { err });
    }
    (worst, agrees)
}

/// What one device measured of a case: its per-repeat host wall time, the device time of the same steps
/// when the device reports it, and the realized plan, counters and reference comparison.
struct Measured {
    wall_us: Vec<f64>,
    /// Sum of the step's dispatch durations on the device clock, per repeat; `None` when the device
    /// reports no timestamps (`DeviceTime::Unknown`), never a fabricated zero.
    device_us: Option<Vec<f64>>,
    plan: Vec<String>,
    worst: f64,
    agrees: bool,
    memory: Vec<Value>,
    recordings: u64,
    replays: u64,
}

fn measure<D>(
    device: D,
    bank: &Value,
    prepared: &Prepared,
    ops: Operands,
) -> Result<Measured, String>
where
    D: Device + 'static,
{
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));

    let mut builder = WeightStore::builder();
    for (name, tensor) in &prepared.consts {
        let dense = DenseWeight::try_new(
            tensor.dtype(),
            tensor.shape().to_vec(),
            StoredBytes::from(tensor.view().bytes().to_vec()),
        )
        .map_err(|e| format!("const {name}: {e}"))?;
        builder
            .insert(name.as_str(), WeightEntry::Dense(dense))
            .map_err(|e| format!("const {name}: {e}"))?;
    }
    let store = builder.build();
    let mut weights = ConstWeights::new(&store);
    let graphs: Vec<Graph> = prepared.graphs.iter().map(|g| weights.graph(g)).collect();
    let exe = exec
        .load_weights(Arc::new(store.clone()), Arc::new(weights.map()))
        .map_err(|e| format!("load_weights: {e}"))?;
    let mut entries = Vec::new();
    let mut plan = Vec::new();
    for g in &graphs {
        let program = staged(g, target, ops.policy())?;
        plan = plan_of(&program);
        entries.push(
            exec.add_entry(exe, &program)
                .map_err(|e| format!("add_entry: {e}"))?,
        );
    }

    let timing = &bank["timing_policy"];
    let warmup = timing["warmup"].as_u64().ok_or("timing_policy.warmup")? as usize;
    let repeats = timing["repeats"].as_u64().ok_or("timing_policy.repeats")? as usize;
    // One step: (wall microseconds, device microseconds when the device timed it, the read-back output).
    let mut step = |index: usize| -> Result<(f64, Option<f64>, HostTensor), String> {
        let entry = entries[index % entries.len()];
        let mut inputs = StepInputs::new();
        for (key, shape, bytes) in &prepared.slots {
            let words = HostView::new(poot_tensor::DType::I32, bytes.len() / 4, bytes)
                .map_err(|e| format!("slot input: {e}"))?;
            inputs.push(key.clone(), shape, words);
        }
        let started = Instant::now();
        let mut outputs = exec
            .step(exe, entry, &inputs, &mut NoSync)
            .map_err(|e| format!("step: {e}"))?;
        let device_us = match outputs.device_time() {
            DeviceTime::Measured(measured) => measured
                .sum_of_dispatch_durations
                .map(|sum| sum.as_secs_f64() * 1e6),
            DeviceTime::Unknown => None,
        };
        let output = outputs.to_host().map_err(|e| format!("read: {e}"))?;
        Ok((started.elapsed().as_secs_f64() * 1e6, device_us, output))
    };

    // The first pass over the copies records every entry and checks each copy's output against its own
    // reference; those steps are warmup. Further warmup keeps the rotation going before the timed repeats.
    let mut worst = 0.0f64;
    let mut agrees = true;
    for (copy, want) in prepared.references.iter().enumerate() {
        let (_, _, output) = step(copy)?;
        if output.shape() != prepared.out_shape.as_slice() {
            return Err(format!(
                "output shape {:?}, expected {:?}",
                output.shape(),
                prepared.out_shape
            ));
        }
        let got = output.as_f32().ok_or("the output is not F32")?;
        let (err, ok) = reference_check(got, want, ops.tolerance());
        worst = worst.max(err);
        agrees &= ok;
    }
    for index in 0..warmup {
        step(prepared.references.len() + index)?;
    }
    let (mut wall_us, mut device_samples) =
        (Vec::with_capacity(repeats), Vec::with_capacity(repeats));
    for index in 0..repeats {
        let (wall, device, _) = step(prepared.references.len() + warmup + index)?;
        wall_us.push(wall);
        device_samples.push(device);
    }
    // A device either times every step or none: a partly timed run would mix two populations in one cell.
    let device_us = if device_samples.iter().all(Option::is_some) {
        Some(device_samples.into_iter().flatten().collect())
    } else if device_samples.iter().all(Option::is_none) {
        None
    } else {
        return Err("the device timed some steps and not others".into());
    };

    let stats = exec.stats();
    let memory: Vec<Value> = stats
        .memory
        .iter()
        .map(|(role, m)| json!({ "role": format!("{role:?}"), "live_bytes": m.live_bytes, "peak_bytes": m.peak_bytes }))
        .collect();
    Ok(Measured {
        wall_us,
        device_us,
        plan,
        worst,
        agrees,
        memory,
        recordings: stats.recordings,
        replays: stats.replays,
    })
}

/// Measures the case twice: wall time on `wall_device` (built without device timing, so its per-dispatch
/// timestamp work never lands in the wall numbers) and device time on `timed_device` (built to report
/// timestamps), each in its own engine. The timed pass's output is checked against the same reference.
fn run_case<D>(
    backend: &str,
    wall_device: D,
    timed_device: D,
    bank: &Value,
    case: &Value,
) -> Result<Value, String>
where
    D: Device + 'static,
{
    let ops = Operands::of(backend, case)?;
    let prepared = prepare(bank, case, ops)?;
    let wall = measure(wall_device, bank, &prepared, ops)?;
    let timed = measure(timed_device, bank, &prepared, ops)?;
    if timed.plan != wall.plan {
        return Err("the timed device realized a different plan than the wall device".into());
    }
    let digest = sha256_digest(wall.plan.join("\n").as_bytes()).as_bytes();
    let exe_path = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let repeat_cell = |metric: &str, samples: &[f64]| {
        json!({
            "repeats": samples
                .iter()
                .enumerate()
                .map(|(slot, micros)| json!({ "slot": slot, "status": "ok", "metrics": { metric: micros } }))
                .collect::<Vec<_>>()
        })
    };
    let mut cells = serde_json::Map::new();
    cells.insert("kernel".into(), repeat_cell("step_us", &wall.wall_us));
    if let Some(device_us) = &timed.device_us {
        cells.insert("device".into(), repeat_cell("device_us", device_us));
    }
    Ok(json!({
        "schema": 1,
        "case": case["id"],
        "backend": backend,
        "role": "baseline",
        "hardware": {
            "backend": backend,
            "device": std::env::var("POOT_BANK_DEVICE").map_err(|_| "POOT_BANK_DEVICE names the measured device")?,
            "driver": std::env::var("POOT_BANK_DRIVER").map_err(|_| "POOT_BANK_DRIVER names its driver")?,
        },
        "binary": exe_path.display().to_string(),
        "observed": prepared.observed,
        "plan": wall.plan,
        "plan_id": hex(&digest),
        "reference": {
            "provenance": "independent-golden",
            "oracle": "f64 loop in the bank runner, sharing no code with the engine under test",
            "max_relative_error": wall.worst.max(timed.worst),
            "tolerance": ops.tolerance(),
            "agrees": wall.agrees && timed.agrees,
        },
        "memory": wall.memory,
        "recordings": wall.recordings,
        "replays": wall.replays,
        "cells": cells,
    }))
}

/// The example's whole `main`: measure the case `--case` of `--bank` on the two devices and write `--out`.
pub(crate) fn run<D>(backend: &str, wall_device: D, timed_device: D) -> Result<(), String>
where
    D: Device + 'static,
{
    let args = parse_args()?;
    let bank: Value = serde_json::from_slice(
        &std::fs::read(&args.bank).map_err(|e| format!("read {}: {e}", args.bank.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", args.bank.display()))?;
    let case = bank["cases"]
        .as_array()
        .and_then(|cases| cases.iter().find(|c| c["id"] == args.case.as_str()))
        .ok_or_else(|| format!("the bank has no case {:?}", args.case))?;
    if case["status"] != "runnable" {
        return Err(format!(
            "case {:?} is {}, not runnable",
            args.case, case["status"]
        ));
    }
    let receipt = run_case(backend, wall_device, timed_device, &bank, case)?;
    if let Some(dir) = Path::new(&args.out).parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    std::fs::write(
        &args.out,
        serde_json::to_string_pretty(&receipt).expect("receipt is JSON") + "\n",
    )
    .map_err(|e| format!("write {}: {e}", args.out.display()))
}
