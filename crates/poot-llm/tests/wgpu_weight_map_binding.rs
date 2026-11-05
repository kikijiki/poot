//! Card 564 on wgpu: qwen2's one body behind `Model`, bound through the weight map (SC-001), its
//! fused and stacked views (SC-004), and a hosted embed gather the binder fills (SC-006). Skips
//! without a Vulkan adapter.

use poot_eval::test_support::weight_map_oracle::eval_mapped;
use poot_executor::{Device, Engine, Executor};
use poot_models::test_support::weight_map::{self as fixture, Projections};
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_wgpu_device::device::WgpuDevice;

fn engine() -> Option<(Box<dyn Executor>, poot_graph_plan::Target)> {
    let device = open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())?;
    let target = device.target();
    Some((Box::new(Engine::new(device)), target))
}

/// SC-001: a BF16 and a Q8_0 qwen2 store bound through the map; each step's logits equal the oracle
/// on the same graph.
///
/// Mutation: pass a `bind` that rebuilds the map with layer 0's Q and K views swapped (both
/// `[64, 64]` at `DIMS`), so the device binds through it while the oracle keeps the model's map; the
/// BF16 prefill goes red at token 1's logits.
#[test]
fn qwen2_binds_through_the_map_and_matches_the_oracle() {
    let Some((mut exec, target)) = engine() else {
        return;
    };
    for projections in [Projections::Bf16, Projections::Q8_0] {
        fixture::check_matches_oracle(
            exec.as_mut(),
            target,
            projections,
            &|g, store, map, slots, state| eval_mapped(g, store, map, slots, state),
            &|map, _| map.clone(),
        );
    }
}

/// SC-001's chunked half: prefill in chunks of 1, 3 and the whole prompt, then decode, equals
/// token-by-token decode at every position.
#[test]
fn chunked_prefill_then_decode_equals_token_by_token_decode() {
    let Some((mut exec, target)) = engine() else {
        return;
    };
    fixture::check_chunked_prefill(exec.as_mut(), target);
}

/// SC-004: Q/K/V viewed by `RowRange` over a fused entry and the gate by a two-part `RowStack` give
/// the separately stored logits, for a BF16 and a Q8_0 store.
#[test]
fn fused_and_stacked_views_equal_separate_storage() {
    let Some((mut exec, target)) = engine() else {
        return;
    };
    for projections in [Projections::Bf16, Projections::Q8_0] {
        fixture::check_fused_views(exec.as_mut(), target, projections);
    }
}

/// POOT-1423: an F32 embedding table is hosted (the binder copies the rows the step's tokens name from the
/// stored table, which is never a device constant) and every step's logits equal the oracle's. The vocabulary
/// is wide enough to name ids past the first rows.
///
/// Mutation: in `poot_executor::binder::gather_rows`, read the row one past the requested id; the logits go red.
#[test]
fn a_hosted_embed_gather_matches_the_oracle() {
    let Some((mut exec, target)) = engine() else {
        return;
    };
    fixture::check_f32_embed_matches_oracle(
        exec.as_mut(),
        target,
        &|g, store, map, slots, state| eval_mapped(g, store, map, slots, state),
    );
}

/// POOT-1017: under a single-buffer limit below the lm_head, legalize splits the head into row
/// chunks, the binder binds each as a row range of the stored head, and a real driver load generates
/// the unsplit run's logits and greedy tokens.
///
/// Mutations: in `poot_executor::binder::chunks`, shift each chunk's rows by one row, or number the
/// chunks in descending order; the logits go red (or the load refuses the chunk set).
#[test]
fn a_split_head_generates_the_unsplit_logits_and_tokens() {
    let Some((mut exec, target)) = engine() else {
        return;
    };
    fixture::check_split_head(exec.as_mut(), target);
}

/// POOT-651 SC-001: a qwen2.5-0.5B-shaped BF16 checkpoint with a tied head decodes to the oracle's logits
/// with BF16-only `Weight` residency (the head and embedding are one buffer, no F32 copy).
///
/// Mutations: (1) in `fold_dense_contractions`, skip the fold for a BF16 weight: `compile` refuses the
/// head's `transpose` or the bound weight mismatches the kernel's element kind. (2) in
/// `Engine::weight_buffer`, skip the residency lookup: the head uploads as a second buffer, `Weight` bytes
/// grow by the head's `vocab * hidden * 2` and the equality goes red.
#[test]
fn the_tied_head_decodes_on_bf16_resident_weights_and_matches_the_oracle() {
    let Some((mut exec, target)) = engine() else {
        return;
    };
    fixture::check_tied_head(exec.as_mut(), target, &|g, store, map, slots, state| {
        eval_mapped(g, store, map, slots, state)
    });
}

/// POOT-1423 SC-005: a dense and a packed gather-only table hosted by token (rows copied at bind time) and by an
/// index the device computes (rows copied at a host sync) give the CPU oracle's output on the real device.
///
/// Mutation: in `poot_executor::binder::gather_rows`, read the row one past the requested id; every case goes red.
#[test]
fn hosted_dense_and_packed_gathers_match_the_oracle_at_bind_and_sync_time() {
    let Some((mut exec, target)) = engine() else {
        return;
    };
    fixture::check_hosted_gathers(exec.as_mut(), target, &|g, store, map, slots, state| {
        eval_mapped(g, store, map, slots, state)
    });
}
