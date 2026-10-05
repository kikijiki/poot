//! Test harness the family modules share: a built model run on the CPU oracle, a legacy tracer's
//! graph evaluated over a transposed copy of the same store (the independent reference of
//! ADR-0101 tiers 1 and 2), Q8_0 and decoded copies of a store, and the checks every family owes
//! (spec 999 SC-003; Card 736 SC-002 to SC-004).

use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_graph_ir::{Graph, OpKind, Slot, Storage, ValidationChannel};
use poot_graph_plan::{WeightFormats, bind_packed_weights};
use poot_load::gguf::{GgufIndex, GgufValue, IdentityNames, read_gguf, write_gguf};
use poot_quant::SourceRole;
use poot_quant::format::WeightFormat;
use poot_quant::weights::{
    AttnRole, DenseWeight, FfnRole, HandleFormat, WeightEntry, WeightMap, WeightRole, WeightStore,
};
use poot_tensor::DType;

use super::standard::oracle::{self, run_step};
use crate::model::{KvLayout, LogitRows, Model, ModelError, Phase, StepShape};
use crate::registry::{FamilyEntry, Fixture, RawConfig, Registry};
use poot_quant::weights::WeightMapError;

/// The cache capacity every family test traces at.
pub(crate) const CAP: usize = 8;

/// Tier 2 (ADR-0101): every element within `1e-4 * (1 + |want|)`; NaN fails.
pub(crate) fn close(got: &[f32], want: &[f32]) {
    let want: Vec<f64> = want.iter().map(|&v| v.into()).collect();
    oracle::assert_matches_f64(got, &want, 1e-4);
}

pub(crate) fn close_all(got: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len(), "state count");
    for (got, want) in got.iter().zip(want) {
        close(got, want);
    }
}

/// A stored entry's logical values: a dense entry decoded, a packed one dequantized.
pub(crate) fn values(entry: &WeightEntry) -> Vec<f32> {
    match entry {
        WeightEntry::Dense(dense) => {
            let mut store = WeightStore::builder();
            store
                .insert("w", WeightEntry::Dense(dense.clone()))
                .unwrap();
            poot_eval::materialize_dense(&store.build(), "w")
                .unwrap()
                .to_f32()
                .unwrap()
                .into_owned()
        }
        WeightEntry::Packed(payload) => {
            let [rows, k] = payload.weight().shape();
            let mut out = vec![0.0; rows * k];
            for (row, chunk) in out.chunks_mut(k).enumerate() {
                payload.decode_row(row, chunk).unwrap();
            }
            out
        }
    }
}

pub(crate) fn f32_entry(shape: Vec<usize>, values: &[f32]) -> WeightEntry {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    WeightEntry::Dense(DenseWeight::try_new(DType::F32, shape, bytes.into()).unwrap())
}

/// The same checkpoint with every entry decoded to F32 (the reference a narrow store is compared
/// with).
pub(crate) fn decoded(store: &WeightStore) -> WeightStore {
    let mut out = WeightStore::builder();
    for (key, entry) in store.iter() {
        out.insert(key.clone(), f32_entry(entry.shape(), &values(entry)))
            .unwrap();
    }
    out.build()
}

/// `store` with each tensor `is_projection` names replaced by a random Q8_0 payload of its shape.
pub(crate) fn q8_0_store(store: &WeightStore, is_projection: impl Fn(&str) -> bool) -> WeightStore {
    let mut out = WeightStore::builder();
    for (i, (key, entry)) in store.iter().enumerate() {
        let entry = if is_projection(key.as_str()) {
            let shape = entry.shape();
            WeightEntry::Packed(Arc::new(poot_test_util::packed::random_payload(
                WeightFormat::Q8_0,
                [shape[0], shape[1]],
                i as u64,
            )))
        } else {
            entry.clone()
        };
        out.insert(key.clone(), entry).unwrap();
    }
    out.build()
}

pub(crate) fn step_shape(tokens: usize, capacity: usize, logits: LogitRows) -> StepShape {
    StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::new(tokens).unwrap(),
        capacity: NonZeroUsize::new(capacity).unwrap(),
        kv: KvLayout::Contiguous,
        logits,
    }
}

/// `entry`'s model over `store`, panicking with the typed error.
pub(crate) fn build(
    entry: &FamilyEntry,
    raw: &RawConfig<'_>,
    store: &WeightStore,
) -> Box<dyn Model> {
    (entry.build)(raw, store).unwrap_or_else(|e| panic!("{}: {e}", entry.family))
}

/// The graph const names of every projection weight a model maps: the attention Q/K/V/O and every
/// feed-forward weight.
pub(crate) fn projection_consts(model: &dyn Model) -> Vec<String> {
    model
        .weights()
        .iter()
        .filter(|(id, _, _)| {
            matches!(
                id.role,
                WeightRole::Attn(
                    AttnRole::Q | AttnRole::K | AttnRole::V | AttnRole::Qkv | AttnRole::O
                ) | WeightRole::Ffn(FfnRole::Gate | FfnRole::Up | FfnRole::Down)
            )
        })
        .map(|(id, _, _)| id.const_name())
        .collect()
}

pub(crate) fn declared_dtype<V: ValidationChannel>(g: &Graph<V>, name: &str) -> Option<DType> {
    g.inputs
        .iter()
        .find(|&&id| g.meta(id).name.as_deref() == Some(name))
        .map(|&id| g.aval(id).dtype)
}

/// What a legacy tracer's graph reads that the store does not hold under the same name: named
/// host constants (its RoPE tables, its ALiBi slopes).
pub(crate) type HostConsts<'a> = &'a [(&'a str, Vec<f32>)];

/// Evaluate a legacy tracer's graph over a transposed copy of `store` (the const map
/// is built here, in the test module). The legacy tracers read HF names, every 2-D weight but the
/// ones in `untransposed` pre-transposed to `[in, out]`, and `host` constants by name.
pub(crate) fn legacy_eval(
    g: &Graph,
    store: &WeightStore,
    host: HostConsts<'_>,
    untransposed: &[&str],
    tokens: &[i32],
    pos: &[i32],
    state: &[Vec<f32>],
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut consts: Vec<(String, Vec<f32>)> = Vec::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        if meta.storage != Storage::Const {
            continue;
        }
        let name = meta.name.clone().unwrap();
        let v = match host.iter().find(|(n, _)| *n == name) {
            Some((_, v)) => v.clone(),
            None => {
                let entry = store
                    .get(&name)
                    .unwrap_or_else(|| panic!("no fixture tensor {name}"));
                let (shape, v) = (entry.shape(), values(entry));
                if shape.len() == 2 && !untransposed.contains(&name.as_str()) {
                    let (rows, cols) = (shape[0], shape[1]);
                    (0..rows * cols)
                        .map(|i| v[(i % rows) * cols + i / rows])
                        .collect()
                } else {
                    v
                }
            }
        };
        consts.push((name, v));
    }
    let consts: Vec<(&str, &[f32])> = consts
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    let zeros: Vec<Vec<f32>> = g
        .state
        .iter()
        .map(|&(si, _)| vec![0.0; g.aval(si).numel()])
        .collect();
    let state: Vec<&[f32]> = if state.is_empty() { &zeros } else { state }
        .iter()
        .map(Vec::as_slice)
        .collect();
    oracle::eval(
        g,
        store,
        &WeightMap::default(),
        &consts,
        &[(Slot::Token, tokens), (Slot::Pos, pos)],
        &state,
    )
}

/// A step's logits and its carried state, as f32.
pub(crate) type StepOutput = (Vec<f32>, Vec<Vec<f32>>);

/// Evaluate a legacy graph on `(tokens, positions, state)`.
pub(crate) type LegacyEval<'a> = dyn Fn(&Graph, &[i32], &[i32], &[Vec<f32>]) -> StepOutput + 'a;

/// A legacy tracer pair and how to evaluate it: the prefill graph of the first `n` tokens (from
/// position 0), the one decode graph, and an evaluator over each.
pub(crate) struct Legacy<'a> {
    pub prefill: &'a dyn Fn(usize) -> Graph,
    pub decode: &'a Graph,
    pub eval: &'a LegacyEval<'a>,
}

/// SC-002 (ADR-0101 tiers 1 and 2, spec 999 SC-003): on `store`, the new body's prefill logits at
/// every position and its carried KV equal the legacy prefill tracer (each prefix's last logits),
/// and three decode steps equal the legacy decode tracer, logits and every KV element, each path
/// carrying its own caches.
pub(crate) fn assert_matches_legacy(model: &dyn Model, store: &WeightStore, legacy: &Legacy<'_>) {
    let vocab = model.config().vocab;
    let prompt = [3, 17, 40, 8, 25];
    let (logits, mut state) = run_step(model, store, Phase::Prefill, CAP, &prompt, 0, &[]);
    let mut legacy_state = Vec::new();
    for n in 1..=prompt.len() {
        let pos: Vec<i32> = (0..n as i32).collect();
        let g = (legacy.prefill)(n);
        let (old, old_state) = (legacy.eval)(&g, &prompt[..n], &pos, &[]);
        close(&logits[(n - 1) * vocab..n * vocab], &old);
        if !g.state.is_empty() {
            legacy_state = old_state;
        }
    }
    if legacy_state.is_empty() {
        // The legacy prefill carries no KV: its caches are the legacy decode over the prompt.
        for (i, &token) in prompt.iter().enumerate() {
            legacy_state = (legacy.eval)(legacy.decode, &[token], &[i as i32], &legacy_state).1;
        }
    }
    close_all(&state, &legacy_state);

    for (i, token) in [11, 30, 2].into_iter().enumerate() {
        let pos = (prompt.len() + i) as i32;
        let (new, new_state) = run_step(model, store, Phase::Decode, CAP, &[token], pos, &state);
        let (old, old_state) = (legacy.eval)(legacy.decode, &[token], &[pos], &legacy_state);
        close(&new, &old);
        close_all(&new_state, &old_state);
        (state, legacy_state) = (new_state, old_state);
    }
}

/// The five largest logits of one step as `(token id, logit)`, largest first.
pub(crate) type Top5 = [(usize, f32); 5];

/// The steps of [`assert_matches_recorded`]: the last prefill position's top five and each of the
/// three decode steps' top five, recorded from the legacy tracers at the base commit `bbd5c3232`.
pub(crate) struct Recorded {
    pub prefill: Top5,
    pub decode: [Top5; 3],
}

fn assert_top5(got: &[f32], want: &Top5, label: &str) {
    let mut ids: Vec<usize> = (0..got.len()).collect();
    ids.sort_by(|&a, &b| got[b].total_cmp(&got[a]));
    let want_ids: Vec<usize> = want.iter().map(|&(id, _)| id).collect();
    assert_eq!(
        &ids[..5],
        &want_ids[..],
        "{label}: the five largest token ids"
    );
    let got_top: Vec<f32> = want.iter().map(|&(id, _)| got[id]).collect();
    let want_top: Vec<f32> = want.iter().map(|&(_, logit)| logit).collect();
    close(&got_top, &want_top);
}

/// SC-002 (ADR-0101 tiers 2 and 3): on `store`, the prompt `[3, 17, 40, 8, 25]` prefilled whole,
/// then decode steps on tokens `[11, 30, 2]`, give the recorded top-five logits at the last
/// prefill position and at each decode step (token ids exact, logits tier 2).
pub(crate) fn assert_matches_recorded(model: &dyn Model, store: &WeightStore, rec: &Recorded) {
    let vocab = model.config().vocab;
    let prompt = [3, 17, 40, 8, 25];
    let (logits, mut state) = run_step(model, store, Phase::Prefill, CAP, &prompt, 0, &[]);
    assert_top5(
        &logits[(prompt.len() - 1) * vocab..],
        &rec.prefill,
        "prefill",
    );
    for (i, token) in [11, 30, 2].into_iter().enumerate() {
        let pos = (prompt.len() + i) as i32;
        let (logits, next) = run_step(model, store, Phase::Decode, CAP, &[token], pos, &state);
        assert_top5(&logits, &rec.decode[i], &format!("decode step {i}"));
        state = next;
    }
}

/// SC-003 (Card 732): prefill through the one body in chunks of 1, 3 and the whole prompt, each
/// chunk continuing from the absolute positions in `Pos`, equals token-by-token decode: logits at
/// every position and the final caches.
pub(crate) fn assert_chunked_prefill_equals_decode(model: &dyn Model, store: &WeightStore) {
    let prompt = [3, 17, 40, 8, 25, 9];
    let (mut want, mut want_state) = (Vec::new(), Vec::new());
    for (i, &token) in prompt.iter().enumerate() {
        let (logits, state) = run_step(
            model,
            store,
            Phase::Decode,
            CAP,
            &[token],
            i as i32,
            &want_state,
        );
        want.extend(logits);
        want_state = state;
    }
    for chunk in [1, 3, prompt.len()] {
        let (mut got, mut state) = (Vec::new(), Vec::new());
        for (c, part) in prompt.chunks(chunk).enumerate() {
            let start = (c * chunk) as i32;
            let (logits, next) = run_step(model, store, Phase::Prefill, CAP, part, start, &state);
            got.extend(logits);
            state = next;
        }
        close(&got, &want);
        close_all(&state, &want_state);
    }
}

/// SC-004's HF half: the BF16 `fixture` store and its Q8_0 copy (each tensor
/// `is_projection` names packed) trace through the one body. The BF16 graph declares every
/// projection at its stored BF16; the Q8_0 graph, after the packed-weight transform fed by the
/// model's map, holds one `PackedDequant` per projection and no projection const. Each equals
/// (tier 2) the same body over its store decoded to F32, prefill and a decode step.
pub(crate) fn assert_bf16_and_q8_0_trace_through_the_packed_transform(
    entry: &FamilyEntry,
    fixture: &Fixture,
    is_projection: impl Fn(&str) -> bool,
) {
    let raw = fixture.raw();
    let prompt = [3, 17, 40, 8];
    for (label, store) in [
        ("bf16", fixture.store.clone()),
        ("q8_0", q8_0_store(&fixture.store, &is_projection)),
    ] {
        let model = build(entry, &raw, &store);
        let projections = projection_consts(&*model);
        let g = model
            .trace(
                Phase::Prefill,
                step_shape(prompt.len(), CAP, LogitRows::All),
            )
            .unwrap();
        assert_packed_binding(&*model, &g, &projections, label == "q8_0", label);

        let reference_store = decoded(&store);
        let reference = build(entry, &raw, &reference_store);
        let (got, state) = run_step(&*model, &store, Phase::Prefill, CAP, &prompt, 0, &[]);
        let (want, want_state) = run_step(
            &*reference,
            &reference_store,
            Phase::Prefill,
            CAP,
            &prompt,
            0,
            &[],
        );
        close(&got, &want);
        close_all(&state, &want_state);
        let (got, _) = run_step(&*model, &store, Phase::Decode, CAP, &[5], 4, &state);
        let (want, _) = run_step(
            &*reference,
            &reference_store,
            Phase::Decode,
            CAP,
            &[5],
            4,
            &want_state,
        );
        close(&got, &want);
    }
}

/// After the packed-weight transform fed by `model`'s map, `g` holds one `PackedDequant` per
/// projection and no projection const when `packed`, and declares each projection at its stored
/// BF16 with no dequant otherwise.
pub(crate) fn assert_packed_binding<V: ValidationChannel + Clone>(
    model: &dyn Model,
    g: &Graph<V>,
    projections: &[String],
    packed: bool,
    label: &str,
) {
    let bound = bind_packed_weights(g, &WeightFormats::from_weight_map(model.weights())).unwrap();
    let dequants = bound
        .eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::PackedDequant { .. }))
        .count();
    for name in projections {
        let declared = declared_dtype(&bound, name);
        if packed {
            assert_eq!(declared, None, "{label}: {name} is still a const");
        } else {
            assert_eq!(declared, Some(DType::BF16), "{label} {name}");
        }
    }
    assert_eq!(
        dequants,
        if packed { projections.len() } else { 0 },
        "{label}"
    );
}

/// SC-003 (Card 736, dfamily F6): `fixture`'s store without `removed` fails `Registry::build`
/// with the weight map's typed missing-weight error naming `removed`, before any trace.
pub(crate) fn assert_missing_weight_is_named(
    registry: &Registry,
    fixture: &Fixture,
    removed: &str,
) {
    let mut store = WeightStore::builder();
    for (key, entry) in fixture.store.iter() {
        if key.as_str() != removed {
            store.insert(key.clone(), entry.clone()).unwrap();
        }
    }
    let store = store.build();
    let err = match registry.build(&fixture.raw(), &store) {
        Err(e) => e,
        Ok(_) => panic!("a store without {removed} built"),
    };
    let crate::registry::LoadModelError::Model(ModelError::Weight {
        source: WeightMapError::Missing { key, .. },
        ..
    }) = &err
    else {
        panic!("expected a missing-weight error for {removed}, got {err}");
    };
    assert_eq!(key.as_str(), removed, "{err}");
}

/// A GGUF checkpoint read back through the real reader: its index and store.
pub(crate) fn read_back(
    kvs: &[(&str, GgufValue)],
    tensors: &[(&str, Vec<u64>, u32, Vec<u8>)],
) -> (GgufIndex, WeightStore) {
    let bytes = write_gguf(kvs, tensors);
    let index = GgufIndex::from_bytes(&bytes).unwrap();
    let store = read_gguf(&index, bytes.as_slice(), &IdentityNames).unwrap();
    (index, store)
}

pub(crate) const GGML_F32: u32 = 0;
pub(crate) const GGML_Q8_0: u32 = 8;

/// One GGUF tensor of `entry` under `name`: a packed Q8_0 entry as its blocks, a dense one as F32.
pub(crate) fn gguf_tensor(name: &str, entry: &WeightEntry) -> (String, Vec<u64>, u32, Vec<u8>) {
    let dims = entry.shape().iter().rev().map(|&d| d as u64).collect();
    let (ty, bytes) = match entry {
        WeightEntry::Packed(payload) => (GGML_Q8_0, payload.bytes(SourceRole::Blocks).to_vec()),
        WeightEntry::Dense(_) => (
            GGML_F32,
            values(entry).iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    };
    (name.to_string(), dims, ty, bytes)
}

/// SC-004's GGUF half: `index` resolves through the registry by its architecture key to
/// `entry`'s family, the built model's projection handles are all packed (with `projections`
/// consts), and its traced graph holds one `PackedDequant` per projection after the packed-weight
/// transform.
pub(crate) fn assert_gguf_resolves_and_traces_packed(
    registry: &Registry,
    entry: &FamilyEntry,
    index: &GgufIndex,
    store: &WeightStore,
    projections: usize,
) -> Box<dyn Model> {
    let raw = RawConfig::Gguf(index);
    assert_eq!(registry.resolve(&raw).unwrap().family, entry.family);
    let model = registry
        .build(&raw, store)
        .unwrap_or_else(|e| panic!("{}: {e}", entry.family));
    let consts = projection_consts(&*model);
    assert_eq!(consts.len(), projections);
    for (id, _, handle) in model.weights().iter() {
        if consts.contains(&id.const_name()) {
            assert!(
                matches!(handle.format, HandleFormat::Packed(_)),
                "{id} is {:?}",
                handle.format
            );
        }
    }
    let g = model
        .trace(Phase::Prefill, step_shape(4, CAP, LogitRows::All))
        .unwrap();
    assert_packed_binding(&*model, &g, &consts, true, "gguf");
    model
}

/// `store` with every tensor whose key contains `norm` replaced by its F32 values plus one: the
/// folded scales a legacy Gemma tracer reads (and a llama.cpp GGUF stores), where the HF checkpoint
/// holds the offsets.
pub(crate) fn with_norm_offsets_folded(store: &WeightStore) -> WeightStore {
    let mut out = WeightStore::builder();
    for (key, entry) in store.iter() {
        let entry = if key.as_str().contains("norm") {
            let folded: Vec<f32> = values(entry).iter().map(|v| v + 1.0).collect();
            f32_entry(entry.shape(), &folded)
        } else {
            entry.clone()
        };
        out.insert(key.clone(), entry).unwrap();
    }
    out.build()
}

/// llama.cpp's GGUF name of an HF dense-decoder tensor; `None` for the head (a GGUF without
/// `output.weight` ties it to the embedding, so a fixture writes it as `output.weight`
/// explicitly through [`gguf_name_with_head`]).
pub(crate) fn gguf_name(hf: &str) -> Option<String> {
    Some(match hf {
        "model.embed_tokens.weight" => "token_embd.weight".to_string(),
        "model.norm.weight" => "output_norm.weight".to_string(),
        "lm_head.weight" => return None,
        _ => {
            let rest = hf.strip_prefix("model.layers.").unwrap();
            let (layer, tensor) = rest.split_once('.').unwrap();
            let tensor = match tensor {
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.q_proj.bias" => "attn_q.bias",
                "self_attn.k_proj.bias" => "attn_k.bias",
                "self_attn.v_proj.bias" => "attn_v.bias",
                "self_attn.q_norm.weight" => "attn_q_norm.weight",
                "self_attn.k_norm.weight" => "attn_k_norm.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                other => panic!("unmapped fixture tensor {other}"),
            };
            format!("blk.{layer}.{tensor}")
        }
    })
}

/// The GGUF name of `hf`, the untied head included (`output.weight`).
pub(crate) fn gguf_name_with_head(hf: &str) -> String {
    gguf_name(hf).unwrap_or_else(|| "output.weight".to_string())
}

/// The GGUF metadata llama.cpp writes for a dense decoder of architecture `arch`, from its HF
/// `config`: the sizes under `{arch}.*`, a token list of the vocabulary's size and the eos id.
pub(crate) fn gguf_dense_kvs(arch: &str, config: &serde_json::Value) -> Vec<(String, GgufValue)> {
    let n = |f: &str| config[f].as_u64().unwrap() as u32;
    let u = GgufValue::U32;
    let mut kvs = vec![
        (
            "general.architecture".to_string(),
            GgufValue::Str(arch.into()),
        ),
        (format!("{arch}.embedding_length"), u(n("hidden_size"))),
        (
            format!("{arch}.feed_forward_length"),
            u(n("intermediate_size")),
        ),
        (format!("{arch}.block_count"), u(n("num_hidden_layers"))),
        (
            format!("{arch}.attention.head_count"),
            u(n("num_attention_heads")),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            u(n("num_key_value_heads")),
        ),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::F32(config["rms_norm_eps"].as_f64().unwrap() as f32),
        ),
        (
            format!("{arch}.context_length"),
            u(n("max_position_embeddings")),
        ),
        (
            format!("{arch}.rope.freq_base"),
            GgufValue::F32(config["rope_theta"].as_f64().unwrap() as f32),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(
                (0..n("vocab_size"))
                    .map(|t| GgufValue::Str(format!("t{t}")))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            u(n("eos_token_id")),
        ),
    ];
    if let Some(d) = config["head_dim"].as_u64() {
        kvs.push((format!("{arch}.attention.key_length"), u(d as u32)));
    }
    kvs
}

/// A GGUF checkpoint of `kvs` and `tensors` read back through the real reader.
pub(crate) fn read_back_owned(
    kvs: &[(String, GgufValue)],
    tensors: &[(String, Vec<u64>, u32, Vec<u8>)],
) -> (GgufIndex, WeightStore) {
    let kvs: Vec<(&str, GgufValue)> = kvs.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = tensors
        .iter()
        .map(|(n, d, t, b)| (n.as_str(), d.clone(), *t, b.clone()))
        .collect();
    read_back(&kvs, &tensors)
}

/// Two graphs are the same equation by equation: the same ops, operands, layer tags and output
/// names, and the same named inputs (spec 999 SC-002).
pub(crate) fn assert_same_graph<V: ValidationChannel, W: ValidationChannel>(
    a: &Graph<V>,
    b: &Graph<W>,
) {
    assert_eq!(a.eqns.len(), b.eqns.len(), "equation count");
    for (i, (x, y)) in a.eqns.iter().zip(&b.eqns).enumerate() {
        assert_eq!(format!("{x:?}"), format!("{y:?}"), "equation {i}");
        assert_eq!(a.meta(x.out).name, b.meta(y.out).name, "equation {i} name");
    }
    let named = |g: &dyn Fn(usize) -> Option<String>, ids: &[usize]| -> Vec<Option<String>> {
        ids.iter().map(|&id| g(id)).collect()
    };
    assert_eq!(
        named(&|id| a.meta(id).name.clone(), &a.inputs),
        named(&|id| b.meta(id).name.clone(), &b.inputs),
        "named inputs"
    );
}
