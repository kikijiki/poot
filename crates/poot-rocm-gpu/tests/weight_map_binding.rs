//! Card 564 on ROCm: qwen2's one body behind `Model`, bound through `WeightSource::Map` (SC-001),
//! and its fused and stacked views (SC-004). Skips without a device unless the lane requires it.

use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::weight_map::{self as fixture, Projections};
use poot_rocm_gpu::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_test_util::device_skip::open_or_skip;
use poot_test_util::weight_map_oracle::eval_mapped;

fn engine() -> Option<(Box<dyn Executor>, poot_graph_plan::Target)> {
    let device = open_or_skip(DeviceBackend::Rocm, RocmDevice::new())?;
    let target = device.target();
    Some((Box::new(Engine::new(device)), target))
}

/// SC-001: a BF16 and a Q8_0 qwen2 store bound through the map; each step's logits equal the oracle
/// on the same graph.
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
