//! Shared CPU-eval helpers of the equivalence tests: one step of a traced family graph over the
//! model's own weights, state carried by the caller.
//! Each test binary uses a subset of these.
#![allow(dead_code)]

use poot_executor_parity::weight_map::MappedModel;
use poot_graph_ir::{Graph, Slot, SlotKey, ValidationOutputs};
use poot_tensor::HostTensor;
use poot_test_util::weight_map_oracle::eval_mapped;

/// `Slot::Token` and `Slot::Pos` of a `[rows, tokens]` step.
pub fn token_pos(
    rows: usize,
    tokens: usize,
    ids: &[i32],
    pos: &[i32],
) -> Vec<(SlotKey, HostTensor)> {
    vec![
        (
            SlotKey::new(Slot::Token, None),
            HostTensor::i32(vec![rows, tokens], ids.to_vec()),
        ),
        (
            SlotKey::new(Slot::Pos, None),
            HostTensor::i32(vec![rows, tokens], pos.to_vec()),
        ),
    ]
}

/// The two slot maps of a paged step.
pub fn slot_maps(
    rows: usize,
    capacity: usize,
    pool_slots: usize,
    read: Vec<i32>,
    write: Vec<i32>,
) -> Vec<(SlotKey, HostTensor)> {
    vec![
        (
            SlotKey::new(Slot::SlotMap, Some("read")),
            HostTensor::i32(vec![rows, capacity], read),
        ),
        (
            SlotKey::new(Slot::SlotMap, Some("write")),
            HostTensor::i32(vec![pool_slots], write),
        ),
    ]
}

/// Zero carried state, in `g.state` order.
pub fn zero_state(g: &Graph<ValidationOutputs>) -> Vec<HostTensor> {
    g.state
        .iter()
        .map(|&(input, _)| HostTensor::zeros(g.aval(input).shape.clone()))
        .collect()
}

/// One CPU-oracle step of `g` over `m`'s weights: its logits and the carried state.
pub fn step(
    m: &MappedModel,
    g: &Graph<ValidationOutputs>,
    slots: &[(SlotKey, HostTensor)],
    state: &[HostTensor],
) -> (HostTensor, Vec<HostTensor>) {
    eval_mapped(g, &m.store, &m.map, slots, state)
}

/// Spec 023 sub-step 1 / card 190: the per-layer K/V cache and the final-position logits a 5-token
/// prompt leaves are the same whether it is prefilled in one forward (a) or decoded token by token
/// over the same capacity (b), within `tol` (both are f32 CPU eval of the same ops in a different
/// composition order). `layers` is the model's layer count: the caches are K and V per layer.
pub fn assert_prefill_fills_cache_like_token_by_token(m: &MappedModel, layers: usize, tol: f32) {
    use poot_executor_parity::dense::step as step_shape;
    use poot_models::model::{LogitRows, Phase};

    let n = 5usize; // prompt length
    let cap = n + 3; // a few free slots, as a real generation would leave
    let tokens: Vec<i32> = vec![3, 7, 1, 9, 2];
    assert_eq!(tokens.len(), n);

    // (b) token-by-token decode, replayed for pos 0..N.
    let gd = m
        .model
        .trace(Phase::Decode, step_shape(1, 1, cap, LogitRows::Last))
        .unwrap();
    let mut tbt_caches = zero_state(&gd);
    let mut tbt_last_logits = None;
    for pos in 0..n {
        let slots = token_pos(1, 1, &tokens[pos..pos + 1], &[pos as i32]);
        let (logits, next) = step(m, &gd, &slots, &tbt_caches);
        tbt_caches = next;
        tbt_last_logits = Some(logits);
    }
    let tbt_last_logits = tbt_last_logits.unwrap();

    // (a) batched prefill: one forward fills slots [0,N).
    let gp = m
        .model
        .trace(Phase::Prefill, step_shape(1, n, cap, LogitRows::Last))
        .unwrap();
    let positions: Vec<i32> = (0..n as i32).collect();
    let (pf_logits, pf_caches) = step(
        m,
        &gp,
        &token_pos(1, n, &tokens, &positions),
        &zero_state(&gp),
    );

    // both graphs carry 2*layers cache buffers in the same (K,V per layer) order.
    assert_eq!(pf_caches.len(), 2 * layers);
    assert_eq!(pf_caches.len(), tbt_caches.len());

    // every slot, including the unwritten tail which stays zero.
    for (ci, (a, b)) in pf_caches.iter().zip(&tbt_caches).enumerate() {
        assert_eq!(a.shape(), b.shape(), "cache {ci} shape");
        // Cache `ci` differs between batched prefill and token-by-token fill if this panics.
        poot_test_util::assert_close(a.as_f32().unwrap(), b.as_f32().unwrap(), tol);
    }

    // the final-position logits must match too (so the next greedy token is unchanged).
    assert_eq!(pf_logits.shape(), tbt_last_logits.shape(), "logits shape");
    // A panic here means the final-position logits differ.
    poot_test_util::assert_close(
        pf_logits.as_f32().unwrap(),
        tbt_last_logits.as_f32().unwrap(),
        tol,
    );
}
