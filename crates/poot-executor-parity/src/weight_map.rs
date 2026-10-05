//! Card 564's device fixtures: qwen2's one body (Card 732) behind [`Model`], over BF16 or Q8_0
//! fixture stores, bound through [`WeightSource::Map`]. A backend's tests run [`run_steps`] on their
//! own `Executor` and compare with the caller's oracle; nothing here depends on `poot-eval`.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_executor::{ExecError, Executor, NoSync, StepInputs, WeightSource};
use poot_graph_ir::{Graph, Slot, SlotKey, ValidationOutputs};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    StagedProgram, Submission, Target, TargetSet, WeightFormats, bind_packed_weights,
    compile_staged,
};
use poot_models::model::{KvLayout, LogitRows, Model, Phase, StepShape};
use poot_models::registry::{RawConfig, Registry};
use poot_quant::PackedPayload;
use poot_quant::format::WeightFormat;
use poot_quant::weights::{
    AttnRole, DenseWeight, FfnRole, WeightEntry, WeightKey, WeightMap, WeightRole, WeightStore,
    WeightView,
};
use poot_tensor::{DType, HostTensor, HostView};

/// A qwen2 checkpoint's dimensions.
#[derive(Clone, Copy, Debug)]
pub struct Qwen2Dims {
    pub vocab: usize,
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
}

/// A tiny qwen2 whose Q and K projections share one shape (so swapping their views binds) and
/// whose projection widths are multiples of 32 (so a Q8_0 copy is a valid checkpoint).
pub const DIMS: Qwen2Dims = Qwen2Dims {
    vocab: 48,
    hidden: 64,
    inter: 96,
    layers: 2,
    heads: 4,
    kv_heads: 4,
};

/// How a fixture store holds its projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projections {
    Bf16,
    Q8_0,
}

/// A model, the store it was built over, and its weight map.
#[derive(Debug)]
pub struct MappedModel {
    pub model: Box<dyn Model>,
    pub store: Arc<WeightStore>,
    pub map: Arc<WeightMap>,
}

/// Deterministic values in `[-0.5, 0.5)`.
fn random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

fn dense(dtype: DType, shape: Vec<usize>, values: &[f32]) -> WeightEntry {
    let bytes: Vec<u8> = match dtype {
        DType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::BF16 => values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        other => panic!("fixture weights are F32 or BF16, not {other:?}"),
    };
    WeightEntry::Dense(DenseWeight::try_new(dtype, shape, Arc::from(bytes)).unwrap())
}

/// A qwen2 checkpoint of `dims` in HF names: norms near 1, everything else small and centred, the
/// head untied, the embedding stored as `embed`, the projections as `projections`; built into its
/// `Model` through the shipped registry.
pub fn qwen2(dims: Qwen2Dims, projections: Projections, embed: DType) -> MappedModel {
    let Qwen2Dims {
        vocab,
        hidden: h,
        inter,
        layers,
        heads,
        kv_heads,
    } = dims;
    let d = h / heads;
    let (q, kv) = (heads * d, kv_heads * d);
    let mut tensors: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![vocab, h]),
        ("model.norm.weight".into(), vec![h]),
        ("lm_head.weight".into(), vec![vocab, h]),
    ];
    for l in 0..layers {
        let k = |s: &str| format!("model.layers.{l}.{s}");
        tensors.extend([
            (k("input_layernorm.weight"), vec![h]),
            (k("post_attention_layernorm.weight"), vec![h]),
            (k("self_attn.q_proj.weight"), vec![q, h]),
            (k("self_attn.k_proj.weight"), vec![kv, h]),
            (k("self_attn.v_proj.weight"), vec![kv, h]),
            (k("self_attn.o_proj.weight"), vec![h, q]),
            (k("self_attn.q_proj.bias"), vec![q]),
            (k("self_attn.k_proj.bias"), vec![kv]),
            (k("self_attn.v_proj.bias"), vec![kv]),
            (k("mlp.gate_proj.weight"), vec![inter, h]),
            (k("mlp.up_proj.weight"), vec![inter, h]),
            (k("mlp.down_proj.weight"), vec![h, inter]),
        ]);
    }
    let mut store = WeightStore::builder();
    for (seed, (key, shape)) in tensors.into_iter().enumerate() {
        let seed = seed as u64 + 1;
        let n: usize = shape.iter().product();
        let entry = if key.ends_with("_proj.weight") && projections == Projections::Q8_0 {
            WeightEntry::Packed(Arc::new(poot_test_util::packed::random_payload(
                WeightFormat::Q8_0,
                [shape[0], shape[1]],
                seed,
            )))
        } else if key.ends_with("norm.weight") {
            let values: Vec<f32> = random(n, seed).iter().map(|u| 1.0 + 0.2 * u).collect();
            dense(DType::BF16, shape, &values)
        } else {
            let values: Vec<f32> = random(n, seed).iter().map(|u| 0.5 * u).collect();
            let dtype = if key == "model.embed_tokens.weight" {
                embed
            } else {
                DType::BF16
            };
            dense(dtype, shape, &values)
        };
        store.insert(key, entry).unwrap();
    }
    let store = store.build();
    let config = serde_json::json!({
        "model_type": "qwen2",
        "vocab_size": vocab,
        "hidden_size": h,
        "intermediate_size": inter,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false,
        "eos_token_id": vocab - 1,
        "bos_token_id": 1,
    });
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
    };
    let model = Registry::builtin()
        .unwrap()
        .build(&raw, &store)
        .unwrap_or_else(|e| panic!("{e}"));
    let map = Arc::new(model.weights().clone());
    MappedModel {
        model,
        store: Arc::new(store),
        map,
    }
}

/// `entry`'s leading-axis rows `rows`, as a store entry of its own.
fn rows_of(entry: &WeightEntry, rows: std::ops::Range<usize>) -> WeightEntry {
    match entry {
        WeightEntry::Dense(d) => {
            let row_bytes = d.bytes().len() / d.shape()[0];
            let mut shape = d.shape().to_vec();
            shape[0] = rows.len();
            let bytes = &d.bytes().as_slice()[rows.start * row_bytes..rows.end * row_bytes];
            WeightEntry::Dense(DenseWeight::try_new(d.dtype(), shape, Arc::from(bytes)).unwrap())
        }
        WeightEntry::Packed(p) => {
            let rows: Vec<usize> = rows.collect();
            WeightEntry::Packed(Arc::new(PackedPayload::gather_rows(&[p], &rows).unwrap()))
        }
    }
}

/// `parts` stacked along the leading axis, as one store entry.
fn stacked(parts: &[&WeightEntry]) -> WeightEntry {
    match parts[0] {
        WeightEntry::Dense(first) => {
            let mut shape = first.shape().to_vec();
            shape[0] = 0;
            let mut bytes = Vec::new();
            for part in parts {
                let WeightEntry::Dense(d) = part else {
                    panic!("mixed parts")
                };
                shape[0] += d.shape()[0];
                bytes.extend_from_slice(d.bytes().as_slice());
            }
            WeightEntry::Dense(
                DenseWeight::try_new(first.dtype(), shape, Arc::from(bytes)).unwrap(),
            )
        }
        WeightEntry::Packed(_) => {
            let payloads: Vec<&PackedPayload> = parts
                .iter()
                .map(|part| match part {
                    WeightEntry::Packed(p) => p.as_ref(),
                    WeightEntry::Dense(_) => panic!("mixed parts"),
                })
                .collect();
            WeightEntry::Packed(Arc::new(PackedPayload::concat_rows(&payloads).unwrap()))
        }
    }
}

/// `m`'s weights re-stored the way a fused checkpoint holds them: each layer's Q, K and V in one
/// `[q + 2 kv, hidden]` entry viewed by `RowRange`, and its gate projection as two half-row
/// entries viewed by a two-part `RowStack`. The separately stored entries are gone from the store,
/// so the views are the only way to reach those bytes. Every handle equals `m`'s.
pub fn fused(m: &MappedModel) -> (Arc<WeightStore>, Arc<WeightMap>) {
    let stored = |view: &WeightView| -> WeightKey {
        let WeightView::Stored(key) = view else {
            panic!("a family map stores each weight whole")
        };
        key.clone()
    };
    let mut replaced: Vec<WeightKey> = Vec::new();
    let mut added: Vec<(String, WeightEntry)> = Vec::new();
    let mut views: Vec<(poot_quant::weights::WeightId, WeightView)> = Vec::new();
    for (id, view, _) in m.map.iter() {
        let layer = id.layer;
        let view = match (layer, id.role) {
            (Some(l), WeightRole::Attn(AttnRole::Q)) => {
                let keys = [AttnRole::Q, AttnRole::K, AttnRole::V].map(|role| {
                    let other = poot_quant::weights::WeightId::layer(l, WeightRole::Attn(role));
                    stored(m.map.view(other).unwrap())
                });
                let entries: Vec<&WeightEntry> = keys
                    .iter()
                    .map(|k| m.store.get(k.as_str()).unwrap())
                    .collect();
                let fused_key = format!("model.layers.{l}.self_attn.qkv_proj");
                added.push((fused_key.clone(), stacked(&entries)));
                replaced.extend(keys);
                let rows = entries[0].shape()[0];
                WeightView::RowRange {
                    key: fused_key.into(),
                    rows: 0..rows,
                }
            }
            (Some(l), WeightRole::Attn(role @ (AttnRole::K | AttnRole::V))) => {
                let row_count = |role| {
                    let other = poot_quant::weights::WeightId::layer(l, WeightRole::Attn(role));
                    m.map.handle(other).unwrap().shape[0]
                };
                let (q, k) = (row_count(AttnRole::Q), row_count(AttnRole::K));
                let start = if role == AttnRole::K { q } else { q + k };
                WeightView::RowRange {
                    key: format!("model.layers.{l}.self_attn.qkv_proj").into(),
                    rows: start..start + row_count(role),
                }
            }
            (Some(l), WeightRole::Ffn(FfnRole::Gate)) => {
                let key = stored(view);
                let entry = m.store.get(key.as_str()).unwrap();
                let half = entry.shape()[0] / 2;
                let parts = [0..half, half..entry.shape()[0]];
                let keys: Vec<WeightKey> = (0..2)
                    .map(|i| WeightKey::from(format!("model.layers.{l}.mlp.gate_proj.part{i}")))
                    .collect();
                for (k, rows) in keys.iter().zip(parts) {
                    added.push((k.as_str().to_string(), rows_of(entry, rows)));
                }
                replaced.push(key);
                WeightView::RowStack(keys)
            }
            _ => view.clone(),
        };
        views.push((id, view));
    }
    let mut store = WeightStore::builder();
    for (key, entry) in m.store.iter() {
        if !replaced.contains(key) {
            store.insert(key.clone(), entry.clone()).unwrap();
        }
    }
    for (key, entry) in added {
        store.insert(key, entry).unwrap();
    }
    let store = store.build();
    let mut map = WeightMap::builder(&store);
    for (id, view) in views {
        map.map(id, view).unwrap();
    }
    let map = map.build();
    for (id, _, handle) in m.map.iter() {
        assert_eq!(
            map.handle(id),
            Some(handle),
            "{id}: the fused view's handle"
        );
    }
    (Arc::new(store), Arc::new(map))
}

/// One step of `tokens` new tokens from absolute position `start`.
#[derive(Clone, Debug)]
pub struct Step {
    pub phase: Phase,
    pub ids: Vec<i32>,
    pub start: usize,
    shape: [usize; 2],
    ids_bytes: Vec<u8>,
    pos_bytes: Vec<u8>,
}

impl Step {
    pub fn new(phase: Phase, ids: &[i32], start: usize) -> Self {
        let pos: Vec<i32> = (start..start + ids.len()).map(|p| p as i32).collect();
        Self {
            phase,
            ids: ids.to_vec(),
            start,
            shape: [1, ids.len()],
            ids_bytes: ids.iter().flat_map(|v| v.to_le_bytes()).collect(),
            pos_bytes: pos.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    pub fn inputs(&self) -> StepInputs<'_> {
        let n = self.ids.len();
        let mut inputs = StepInputs::new();
        for (slot, bytes) in [(Slot::Token, &self.ids_bytes), (Slot::Pos, &self.pos_bytes)] {
            inputs.push(
                SlotKey::new(slot, None),
                &self.shape,
                HostView::new(DType::I32, n, bytes).unwrap(),
            );
        }
        inputs
    }

    /// The step's slots as host tensors, for an oracle.
    pub fn slots(&self) -> Vec<(SlotKey, HostTensor)> {
        let pos = (self.start..self.start + self.ids.len())
            .map(|p| p as i32)
            .collect();
        vec![
            (
                SlotKey::new(Slot::Token, None),
                HostTensor::i32(self.shape.to_vec(), self.ids.clone()),
            ),
            (
                SlotKey::new(Slot::Pos, None),
                HostTensor::i32(self.shape.to_vec(), pos),
            ),
        ]
    }
}

/// `model`'s graph for `phase` with `tokens` new tokens over a cache of `capacity`, logits at every
/// token, its packed weights bound from `map`.
pub fn trace(
    model: &dyn Model,
    map: &WeightMap,
    phase: Phase,
    tokens: usize,
    capacity: usize,
) -> Graph<ValidationOutputs> {
    let shape = StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::new(tokens).unwrap(),
        capacity: NonZeroUsize::new(capacity).unwrap(),
        kv: KvLayout::Contiguous,
        logits: LogitRows::All,
    };
    let g = model.trace(phase, shape).unwrap();
    bind_packed_weights(&g, &WeightFormats::from_weight_map(map)).unwrap()
}

/// `g` compiled for `target`, single stage and single device, replayed.
pub fn compile(g: &Graph<ValidationOutputs>, target: Target) -> StagedProgram<ValidationOutputs> {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            execution: Submission::Replay,
            fusion: FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .unwrap_or_else(|e| panic!("compile: {e}"))
}

/// The cache capacity every fixture step traces with.
pub const CAPACITY: usize = 8;

/// Load `store` bound through `map` on `exec`, then run `steps` in order over carried state (zero at
/// the start), one entry per distinct (phase, token count), each traced over [`CAPACITY`].
/// `prepare` rewrites each traced graph before it compiles (identity, or a legalize for a fixture
/// cap). Returns each step's logits.
pub fn run_steps(
    exec: &mut dyn Executor,
    target: Target,
    model: &dyn Model,
    store: Arc<WeightStore>,
    map: Arc<WeightMap>,
    steps: &[Step],
    prepare: &dyn Fn(Graph<ValidationOutputs>) -> Graph<ValidationOutputs>,
) -> Result<Vec<Vec<f32>>, ExecError> {
    let exe = exec.load_weights(store, WeightSource::Map(Arc::clone(&map)))?;
    let mut entries = HashMap::new();
    let mut logits = Vec::with_capacity(steps.len());
    for step in steps {
        let key = (step.phase, step.ids.len());
        let entry = match entries.get(&key) {
            Some(&entry) => entry,
            None => {
                let g = prepare(trace(model, &map, step.phase, step.ids.len(), CAPACITY));
                let entry = exec.add_entry(exe, &compile(&g, target))?;
                entries.insert(key, entry);
                entry
            }
        };
        let out = exec
            .step(exe, entry, &step.inputs(), &mut NoSync)?
            .to_host()?;
        logits.push(out.as_f32().expect("f32 logits").to_vec());
    }
    exec.unload(exe)?;
    Ok(logits)
}

/// Token-by-token decode of `ids` from position 0.
pub fn decode_steps(ids: &[i32]) -> Vec<Step> {
    ids.iter()
        .enumerate()
        .map(|(pos, &id)| Step::new(Phase::Decode, &[id], pos))
        .collect()
}

/// `prompt` prefilled in chunks of `chunk` tokens, then `decode` decoded one at a time.
pub fn chunked_steps(prompt: &[i32], chunk: usize, decode: &[i32]) -> Vec<Step> {
    let mut steps: Vec<Step> = prompt
        .chunks(chunk)
        .scan(0, |start, ids| {
            let step = Step::new(Phase::Prefill, ids, *start);
            *start += ids.len();
            Some(step)
        })
        .collect();
    steps.extend(
        decode
            .iter()
            .enumerate()
            .map(|(i, &id)| Step::new(Phase::Decode, &[id], prompt.len() + i)),
    );
    steps
}

/// The caller's CPU oracle for one step of a mapped graph (`poot_test_util::weight_map_oracle::
/// eval_mapped`): the graph, the store and map its weights bind through, the step's slots and the
/// carried state; returns the output and the state outputs.
pub type MappedOracle<'a> = dyn Fn(
        &Graph<ValidationOutputs>,
        &WeightStore,
        &WeightMap,
        &[(SlotKey, HostTensor)],
        &[HostTensor],
    ) -> (HostTensor, Vec<HostTensor>)
    + 'a;

/// ADR-0101 tier 2: every element of `got` within `tol * (1 + |want|)` of `want`; NaN fails.
pub fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: element count");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * (1.0 + w.abs()),
            "{label}: element {i}: got {g}, want {w}"
        );
    }
}

/// The tier-2 tolerance of a device step against the CPU oracle: `TIER2 * (1 + |want|)` per element.
/// ADR-0101 (Backstage `adr/0101.md`, "Decision", item 2, "Tier 2, stated tolerance") derives the
/// tolerance from the program's reassociation class and states no number; these programs are
/// `Tier2Class::NarrowStorage` (`poot-graph-ir/src/analysis/numerics.rs`: BF16 and Q8_0 weights read
/// by `DenseContraction`/`PackedContraction`, plus fused rows and RoPE), the widest class, and no
/// class-to-tolerance table exists in the tree yet, so this stated bound is the row's own.
pub const TIER2: f32 = 1e-3;

/// SC-001: qwen2's one body over a `projections` store, bound through `WeightSource::Map`: a
/// prefill of three tokens and the decode after it each equal the oracle on the same graph, state
/// carried. `bind` is the map the device binds through (the model's own; a mutation swaps views).
pub fn check_matches_oracle(
    exec: &mut dyn Executor,
    target: Target,
    projections: Projections,
    oracle: &MappedOracle<'_>,
    bind: &dyn Fn(&WeightMap, &WeightStore) -> WeightMap,
) {
    let m = qwen2(DIMS, projections, DType::BF16);
    let steps = [
        Step::new(Phase::Prefill, &[3, 1, 4], 0),
        Step::new(Phase::Decode, &[1], 3),
    ];
    let got = run_steps(
        exec,
        target,
        m.model.as_ref(),
        Arc::clone(&m.store),
        Arc::new(bind(&m.map, &m.store)),
        &steps,
        &|g| g,
    )
    .unwrap_or_else(|e| panic!("{projections:?}: {e}"));
    let mut state: Vec<HostTensor> = Vec::new();
    for (step, got) in steps.iter().zip(&got) {
        let g = trace(
            m.model.as_ref(),
            &m.map,
            step.phase,
            step.ids.len(),
            CAPACITY,
        );
        if state.is_empty() {
            state = g
                .state
                .iter()
                .map(|&(input, _)| HostTensor::zeros(g.aval(input).shape.clone()))
                .collect();
        }
        let (want, next) = oracle(&g, &m.store, &m.map, &step.slots(), &state);
        state = next;
        assert_close(
            got,
            want.as_f32().expect("f32 logits"),
            TIER2,
            &format!("{projections:?} {:?} at {}", step.phase, step.start),
        );
    }
}

/// Every logits row of `logits` (one `[1, tokens, vocab]` block per step), in token order.
fn token_rows(logits: &[Vec<f32>], vocab: usize) -> Vec<Vec<f32>> {
    logits
        .iter()
        .flat_map(|step| step.chunks(vocab).map(<[f32]>::to_vec))
        .collect()
}

/// SC-001 (R-562-6, Card 732's device half): a prompt prefilled in chunks of 1, of 3 and whole,
/// then decoded, gives every token's logits equal to token-by-token decode of the same sequence.
pub fn check_chunked_prefill(exec: &mut dyn Executor, target: Target) {
    let m = qwen2(DIMS, Projections::Bf16, DType::BF16);
    let (prompt, decode) = ([3, 1, 4, 1, 5, 9], [2, 6]);
    let run = |exec: &mut dyn Executor, steps: &[Step]| {
        let logits = run_steps(
            exec,
            target,
            m.model.as_ref(),
            Arc::clone(&m.store),
            Arc::clone(&m.map),
            steps,
            &|g| g,
        )
        .unwrap();
        token_rows(&logits, DIMS.vocab)
    };
    let sequence: Vec<i32> = prompt.iter().chain(&decode).copied().collect();
    let reference = run(exec, &decode_steps(&sequence));
    for chunk in [1, 3, prompt.len()] {
        let got = run(exec, &chunked_steps(&prompt, chunk, &decode));
        assert_eq!(got.len(), reference.len());
        for (position, (got, want)) in got.iter().zip(&reference).enumerate() {
            assert_close(
                got,
                want,
                TIER2,
                &format!("chunk {chunk}, position {position}"),
            );
        }
    }
}

/// SC-004: the fused store (Q/K/V by `RowRange` over one `[q + 2 kv, hidden]` entry, the gate by a
/// two-part `RowStack`) gives bit-identical logits to the separately stored one: the same graph over
/// the same bytes.
pub fn check_fused_views(exec: &mut dyn Executor, target: Target, projections: Projections) {
    let m = qwen2(DIMS, projections, DType::BF16);
    let (store, map) = fused(&m);
    let steps = [
        Step::new(Phase::Prefill, &[3, 1, 4], 0),
        Step::new(Phase::Decode, &[1], 3),
    ];
    let run = |exec: &mut dyn Executor, store, map| {
        run_steps(exec, target, m.model.as_ref(), store, map, &steps, &|g| g)
            .unwrap_or_else(|e| panic!("{projections:?}: {e}"))
    };
    let separate = run(exec, Arc::clone(&m.store), Arc::clone(&m.map));
    let viewed = run(exec, store, map);
    for (step, (got, want)) in viewed.iter().zip(&separate).enumerate() {
        let got: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
        let want: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
        assert!(
            got == want,
            "{projections:?} step {step}: the viewed store's logits differ from separate storage"
        );
    }
}

/// POOT-1017: the buffer limit of [`check_split_head`]: under the BF16 head (`[48, 64]`, 6144 bytes),
/// the embed table, and every projection (64 x 64 BF16 is 8192 bytes), so legalize splits the head (and the rest) into
/// row chunks and hosts the embed gather.
pub const SPLIT_LIMIT: u64 = 4096;

/// POOT-1017: the tiny qwen2's lm_head exceeds the target's single-buffer limit, so legalize splits it
/// (and every other weight over the limit) into `<name>.chunkN` consts the binder binds as row ranges of
/// the stored weight. A real driver load of that graph - prefill then decode through the production
/// `load_weights`/`add_entry`/`step` path - gives the unsplit run's logits and greedy tokens.
pub fn check_split_head(exec: &mut dyn Executor, target: Target) {
    // An F32 embedding: its hosted rows are an F32 slot, so no kernel computes in bf16 on the embed path.
    let m = qwen2(DIMS, Projections::Bf16, DType::F32);
    let steps = [
        Step::new(Phase::Prefill, &[3, 1, 4], 0),
        Step::new(Phase::Decode, &[1], 3),
    ];
    let mut caps = target.caps;
    caps.max_buffer_bytes = SPLIT_LIMIT;
    let split = |g: Graph<ValidationOutputs>| {
        let g = poot_graph_plan::legalize(&g, &caps, &poot_graph_plan::CompileLimits::STANDARD)
            .unwrap_or_else(|e| panic!("legalize under a {SPLIT_LIMIT}-byte limit: {e}"));
        let head_chunks = g
            .consts
            .iter()
            .filter(|&&id| {
                g.meta(id)
                    .name
                    .as_deref()
                    .is_some_and(|n| n.starts_with("w.head.chunk"))
            })
            .count();
        assert!(head_chunks >= 2, "the head splits ({head_chunks} chunks)");
        for &id in &g.consts {
            let bytes = g.aval(id).numel() * g.aval(id).dtype.byte_size();
            assert!(
                bytes as u64 <= SPLIT_LIMIT,
                "{:?} ({bytes} bytes) fits the limit",
                g.meta(id).name
            );
        }
        g
    };
    let run = |exec: &mut dyn Executor, prepare: &dyn Fn(_) -> _| {
        run_steps(
            exec,
            target,
            m.model.as_ref(),
            Arc::clone(&m.store),
            Arc::clone(&m.map),
            &steps,
            prepare,
        )
        .unwrap_or_else(|e| panic!("{e}"))
    };
    let whole = run(exec, &|g| g);
    let chunked = run(exec, &split);
    let greedy = |logits: &[f32]| {
        logits
            .chunks(DIMS.vocab)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(token, _)| token)
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    for (step, (got, want)) in chunked.iter().zip(&whole).enumerate() {
        assert_close(got, want, TIER2, &format!("split head, step {step}"));
        assert_eq!(
            greedy(got),
            greedy(want),
            "split head, step {step}: greedy tokens"
        );
    }
}
