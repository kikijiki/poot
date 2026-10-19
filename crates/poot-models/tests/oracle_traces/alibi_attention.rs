//! The BLOOM family's `Model::trace` decode (ALiBi attention): the right graph structure (no RoPE, the bias an
//! in-graph computation) and a finite output on the CPU oracle. (Moved from
//! `poot-eval/src/tests/alibi_attention.rs`; the hand-bias attention rows stay with the oracle.)

use poot_graph_ir::ComputedConst;
use poot_graph_ir::Storage;
use poot_models::model::{LogitRows, Model};
use poot_models::registry::Registry;
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::DType;
use poot_test_util::{fill, seed_of};

use super::qwen2_pipeline::{eval_step, trace_step};

/// The BLOOM family's decode (ALiBi attention, no RoPE): it declares no RoPE table (card 550:
/// the per-head bias is a graph computation, `alibi_mask_from_pos`, over `Slot::Pos` and the family's `AlibiSlopes`
/// computed constant), and it evaluates to finite logits of the vocabulary's shape.
#[test]
fn bloom_alibi_decode_skips_rope() {
    let (model, cap) = (bloom_f32_model(), 8usize);
    let vocab = model.config().vocab;
    let g = trace_step(&*model, (1, 1, cap), None, LogitRows::Last);

    // FR-002: no rope tables declared on the ALiBi path.
    for id in g.inputs() {
        let m = &g.values()[*id];
        if let Storage::Const = m.storage {
            let name = m.name.as_deref().unwrap_or("");
            assert!(
                !name.contains("rope"),
                "an ALiBi decode must not declare rope tables, found {name}"
            );
        }
    }

    // The widened per-head bias is a graph computation over Slot::Pos ([1,1]) and the slopes.
    assert!(
        g.inputs().iter().any(|id| matches!(
            g.values()[*id].storage,
            Storage::Computed(ComputedConst::AlibiSlopes { .. })
        )),
        "an ALiBi decode graph must read the per-head slopes"
    );

    let out = eval_step(&g, &[2], &[3], None, None).logits;
    assert_eq!(out.shape(), vec![1, 1, vocab]);
    for &v in out.as_f32().unwrap().iter() {
        assert!(v.is_finite(), "alibi logit {v} is not finite");
    }
}

/// The BLOOM fixture checkpoint with every tensor held as F32 (deterministic name-seeded values), built through the
/// registry: a graph whose consts the F32 name-seeded fill of [`eval_step`] binds.
fn bloom_f32_model() -> Box<dyn Model> {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|e| e.family.as_str() == "bloom")
        .expect("the bloom family is registered");
    let fixture = (entry.fixture)();
    let mut store = WeightStore::builder();
    for (key, tensor) in fixture.store.iter() {
        let shape = tensor.shape();
        let n: usize = shape.iter().product();
        let values: Vec<f32> = fill(n, seed_of(key.as_str()))
            .iter()
            .map(|v| v * 0.1)
            .collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense = DenseWeight::try_new(DType::F32, shape, StoredBytes::from(bytes)).unwrap();
        store
            .insert(key.clone(), WeightEntry::Dense(dense))
            .unwrap();
    }
    registry
        .build(&fixture.raw(), &store.build())
        .unwrap_or_else(|e| panic!("{e}"))
}
