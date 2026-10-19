//! Card 564 on wgpu: qwen2's one body behind `Model`, bound through the weight map (SC-001), its
//! fused and stacked views (SC-004), and a hosted embed gather the binder fills (SC-006). Skips
//! without a Vulkan adapter.

use poot_eval::test_support::weight_map_oracle::eval_mapped;
use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::assert_close;
use poot_graph_ir::Slot;
use poot_graph_plan::test_support::gpu_caps;
use poot_models::model::Phase;
use poot_models::test_support::weight_map::{self as fixture, DIMS, Projections, Qwen2Dims, Step};
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_tensor::DType;
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

/// An F32 embedding is the one tensor over a cap between it and the BF16 head, so legalize hosts
/// its gather and nothing else.
const HOSTED: Qwen2Dims = Qwen2Dims { vocab: 256, ..DIMS };

/// SC-006: with a fixture cap that makes legalize host the embed gather as `Slot::TokenEmbed`, the
/// binder fills it from the step's tokens and the logits equal the unhosted run.
#[test]
fn a_hosted_embed_gather_equals_the_device_gather() {
    let Some((mut exec, target)) = engine() else {
        return;
    };
    let m = fixture::qwen2(HOSTED, Projections::Bf16, DType::F32);
    let mut caps = gpu_caps(&target);
    caps.max_buffer_bytes = (HOSTED.vocab * HOSTED.hidden * 4 - 1) as u64;
    let hosted = |g| {
        let g = poot_graph_plan::legalize(&g, &caps, &poot_graph_plan::CompileLimits::STANDARD)
            .unwrap();
        assert!(
            g.slots().iter().any(|&(_, slot)| slot == Slot::TokenEmbed),
            "the fixture cap hosts the embed gather"
        );
        g
    };
    let steps = [
        Step::new(Phase::Prefill, &[200, 1, 77], 0),
        Step::new(Phase::Decode, &[255], 3),
    ];
    let run = |exec: &mut dyn Executor, prepare: &dyn Fn(_) -> _| {
        fixture::run_steps(
            exec,
            target,
            m.model.as_ref(),
            m.store.clone(),
            m.map.clone(),
            &steps,
            prepare,
        )
        .unwrap()
    };
    let device_gather = run(exec.as_mut(), &|g| g);
    let host_gather = run(exec.as_mut(), &hosted);
    let tolerance = host_gather.class.max(device_gather.class).tolerance();
    for (step, (got, want)) in host_gather
        .logits
        .iter()
        .zip(&device_gather.logits)
        .enumerate()
    {
        assert_close(got, want, tolerance, &format!("hosted embed, step {step}"));
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
