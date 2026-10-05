//! Card 540b review (`reviews/540b.md`): a confirmed regression - a *tied* checkpoint that still
//! ships an on-disk `lm_head.weight` (a real shape some quantization tools' conversions produce; the
//! real-checkpoint repro is `coherence_private_tests::awq_safetensors_resident_packed_decode_on_gpu`,
//! qwen2.5-0.5b-awq) used to have its correct, embedding-derived transposed `lm_head.weight` silently
//! overwritten by the generic per-tensor loop reading the redundant on-disk entry untransposed. This
//! is the small synthetic fixture for this regression, so the regression does not depend on a real
//! checkpoint's on-disk layout: `vocab != hidden`, so a wrong (untransposed) result is also a wrong
//! *shape*, and the on-disk redundant tensor's values are deliberately unrelated to the embedding's,
//! so a values-only check could not pass by coincidence either.

use super::super::weights::build_weights;
use crate::architectures::qwen2_hf_config::qwen2_config_from_hf;
use poot_load::Qwen2HfConfig;
use poot_quant::weights::WeightStore;

fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

/// A minimal in-memory F32 safetensors archive with exactly the two tensors this fixture needs.
fn safetensors_archive(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let bytes = f32_bytes(values.iter().copied());
        let start = data.len();
        data.extend_from_slice(&bytes);
        header.insert(
            (*name).to_string(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut archive = Vec::with_capacity(8 + header.len() + data.len());
    archive.extend_from_slice(&(header.len() as u64).to_le_bytes());
    archive.extend_from_slice(&header);
    archive.extend_from_slice(&data);
    archive
}

/// `tie_word_embeddings: true`, no MoE/quant fields; `build_weights` only reads the fields
/// `qwen2_config_from_hf`/its own `hf.is_X()` family checks touch.
fn tied_hf_config(vocab: usize, hidden: usize) -> Qwen2HfConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "qwen2",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": hidden,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": 16,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }))
    .expect("minimal qwen2 hf config")
}

/// Card 540b review: a tied model whose store still holds a redundant on-disk `lm_head.weight` (a
/// different value pattern *and* the checkpoint's un-transposed `[vocab, hidden]` shape, deliberately
/// distinguishable from the correct `[hidden, vocab]` transposed embedding) must end up with the
/// transposed-embedding `lm_head.weight`, not the raw on-disk one. Mutation (recorded here, never left
/// in the tree): reverting the drain back to `if !tied { take_dense(...) }` (only draining
/// `"lm_head.weight"` from the store in the untied case) reproduces the original regression - red, but
/// via the *other* half of the same fix rather than a silently wrong value: with the drain gap
/// reopened, the redundant on-disk entry survives to the generic loop, which tries to insert
/// `"lm_head.weight"` a second time and `insert_once` refuses it as a typed error (`build_weights:
/// lm_head.weight was written twice`) instead of the old silent `HashMap::insert` overwrite. Restoring
/// the unconditional drain makes it green again.
#[test]
fn tied_model_drains_a_redundant_on_disk_lm_head_instead_of_letting_it_overwrite() {
    let (vocab, hidden) = (4usize, 3usize);
    // embed[v][h] = v * 10 + h, so the correct transposed lm_head[h][v] = v * 10 + h too (just
    // reshaped/transposed), distinct from the redundant on-disk tensor below.
    let embed: Vec<f32> = (0..vocab * hidden)
        .map(|i| {
            let (v, h) = (i / hidden, i % hidden);
            (v * 10 + h) as f32
        })
        .collect();
    // A redundant on-disk lm_head.weight: same on-disk shape a real checkpoint would ship
    // ([vocab, hidden], untransposed), but every value is 999.0 - nothing a correct transpose of
    // `embed` could ever produce, so any leakage of this tensor into the result is unambiguous.
    let redundant_lm_head = vec![999.0f32; vocab * hidden];
    let archive = safetensors_archive(&[
        (
            "model.embed_tokens.weight",
            vec![vocab, hidden],
            embed.clone(),
        ),
        ("lm_head.weight", vec![vocab, hidden], redundant_lm_head),
    ]);
    let mut store: WeightStore =
        poot_load::safetensors::load_weight_store_bytes(&archive).expect("parse synthetic archive");

    let hf = tied_hf_config(vocab, hidden);
    let cfg = qwen2_config_from_hf(&hf);
    let (w, _) = build_weights(&mut store, &cfg, &hf).expect("build_weights");

    let lm_head = w
        .get("lm_head.weight")
        .expect("lm_head.weight must be present for a tied model");
    assert_eq!(
        lm_head.as_host().expect("dense weight").shape(),
        vec![hidden, vocab],
        "lm_head.weight must be the transposed [hidden, vocab] embedding, not the on-disk \
         [vocab, hidden] redundant tensor"
    );
    for h in 0..hidden {
        for v in 0..vocab {
            let expected = (v * 10 + h) as f32;
            let got = lm_head.as_host().expect("dense weight").as_f32().unwrap()[h * vocab + v];
            assert_eq!(
                got, expected,
                "lm_head.weight[{h}][{v}] = {got}, expected the embedding-derived {expected} \
                 (a redundant on-disk lm_head.weight must never survive into the weight map for a \
                 tied model)"
            );
        }
    }

    // The store itself must not still hold the redundant entry: `build_weights` takes ownership of
    // every store entry it visits (draining, card 540b SC-001), so a leftover `lm_head.weight` here
    // would mean this fixture failed to exercise the drain path at all.
    assert!(
        !store.contains("lm_head.weight"),
        "lm_head.weight must be drained from the store even when tied"
    );
}

/// The untied counterpart: an on-disk `lm_head.weight` with `tie_word_embeddings: false` still ends
/// up transposed in the weight map (unchanged behavior; guards against the fix overcorrecting into
/// dropping the untied case's own legitimate on-disk lm_head).
#[test]
fn untied_model_still_gets_its_own_transposed_lm_head() {
    let (vocab, hidden) = (4usize, 3usize);
    let embed: Vec<f32> = (0..vocab * hidden).map(|i| i as f32).collect();
    // The untied lm_head is its own tensor, independent of the embedding; use a distinct pattern.
    let lm_head_on_disk: Vec<f32> = (0..vocab * hidden).map(|i| 100.0 + i as f32).collect();
    let archive = safetensors_archive(&[
        ("model.embed_tokens.weight", vec![vocab, hidden], embed),
        (
            "lm_head.weight",
            vec![vocab, hidden],
            lm_head_on_disk.clone(),
        ),
    ]);
    let mut store: WeightStore =
        poot_load::safetensors::load_weight_store_bytes(&archive).expect("parse synthetic archive");

    let mut hf = tied_hf_config(vocab, hidden);
    hf.tie_word_embeddings = false;
    let cfg = qwen2_config_from_hf(&hf);
    let (w, _) = build_weights(&mut store, &cfg, &hf).expect("build_weights");

    let lm_head = w.get("lm_head.weight").expect("lm_head.weight present");
    assert_eq!(
        lm_head.as_host().expect("dense weight").shape(),
        vec![hidden, vocab]
    );
    for h in 0..hidden {
        for v in 0..vocab {
            let expected = lm_head_on_disk[v * hidden + h];
            assert_eq!(
                lm_head.as_host().expect("dense weight").as_f32().unwrap()[h * vocab + v],
                expected
            );
        }
    }
}
