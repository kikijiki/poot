//! POOT-1327: what Qwen3.8-Flash-Next UD-Q2_K_XL (arch `qwen4exp`, 78.87 GB over three shards) allocates on an
//! RTX PRO 6000 Blackwell (96 GB), planned without a GPU and without a tensor byte.
//!
//! The checkpoint is the real tensor directory and metadata as three sparse shard files
//! ([`poot_models::test_support::flash_next`]); the model is the registry's, loaded through
//! `ModelHandle::load`; the plan is the production `prepare` over a recording PTX device
//! ([`Planner`]), so the allocations are the ones the engine would make. Run with `--nocapture` for the
//! report: per-role totals and the biggest allocations at each context.
//!
//! Mutation (`the_budget_refuses_flash_next_before_any_allocation`): make `Footprint::largest` return the
//! smallest allocations first; the first-named allocation is no longer the per-layer embedding table.

use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use poot_graph_plan::{Target, multi_device::topology::DeviceId, staged::TargetSet};
use poot_llm::driver::capacity::{
    AdmitError, Budget, CapacityRefusal, DeviceObservation, Footprint, Ledger, Observation,
    Planner, SUPPORT_RESERVE_BYTES, chunk_candidates, fit_prefill_chunk,
};
use poot_llm::driver::placement::Placement;
use poot_llm::driver::{
    DriverOptions, Head, Layout, LoadOptions, ModelHandle, PreparedSetLimits, ServingShapes, Warm,
};
use poot_models::registry::Registry;
use poot_models::test_support::flash_next::SparseSet;
use poot_runtime_common::BufferRole;
use poot_target::{Backend, DeviceCaps};

const GIB: f64 = (1u64 << 30) as f64;

/// `nvidia-smi`'s total of an RTX PRO 6000 Blackwell Server Edition (97,887 MiB). The pod's `cuMemGetInfo`
/// is the truth; this is the figure the card quotes as "96 GB".
const RTX_PRO_6000_BYTES: u64 = 97_887 << 20;

/// The PTX target of that card: the default Ampere-class descriptor with the card's memory. The tile and
/// arena choices of the plan do not read the tensor-core generation, the footprint does not depend on it.
fn rtx_pro_6000() -> TargetSet {
    let caps = DeviceCaps {
        max_buffer_bytes: RTX_PRO_6000_BYTES,
        memory_budget_bytes: Some(RTX_PRO_6000_BYTES),
        ..DeviceCaps::ptx_default()
    };
    TargetSet::single(
        DeviceId(0),
        Target::Gpu {
            backend: Backend::Nvptx,
            caps,
        },
    )
}

fn handle() -> (Arc<ModelHandle>, SparseSet) {
    let set = SparseSet::create();
    let registry = Registry::builtin().unwrap();
    let handle = ModelHandle::load(&set.paths[0], &registry, &LoadOptions::default())
        .unwrap_or_else(|e| panic!("{e}"));
    (Arc::new(handle), set)
}

/// What the benchmark runner prepares for one cell: capacity `isl + osl`, one row, the prompt's own pieces warmed.
fn cell(isl: usize, osl: usize, chunk: usize) -> (DriverOptions, [usize; 1]) {
    let capacity = NonZeroUsize::new(isl + osl).unwrap();
    let options = DriverOptions {
        trace: poot_llm::driver::serving_trace_limits(capacity),
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(64).unwrap(),
            max_retained_bytes: NonZeroU64::new(8 << 30).unwrap(),
        },
        ..DriverOptions::generation(capacity, NonZeroUsize::new(chunk).unwrap())
    };
    (options, [isl])
}

fn shapes(prompts: &[usize]) -> ServingShapes<'_> {
    ServingShapes {
        layout: Layout::Contiguous,
        rows: &[NonZeroUsize::MIN],
        heads: &[Head::GREEDY],
        warm: Warm::Prompts(prompts),
        windows: &[],
        adapters: &[],
        chain: None,
    }
}

fn plan(handle: &Arc<ModelHandle>, isl: usize, osl: usize, chunk: usize) -> Footprint {
    let (options, prompts) = cell(isl, osl, chunk);
    Planner::new(Arc::clone(handle), &rtx_pro_6000(), Placement::Single)
        .unwrap()
        .plan(options, &shapes(&prompts), &[])
        .unwrap_or_else(|e| panic!("{e}"))
        .footprint
}

/// An idle card with `free` bytes free, as the runtime reports it.
fn observe(free: u64) -> Observation {
    Observation {
        devices: vec![DeviceObservation {
            device: DeviceId(0),
            target: rtx_pro_6000().devices()[0].1,
            free_bytes: Some(free),
            held_bytes: 0,
        }],
    }
}

fn report(label: &str, footprint: &Footprint) {
    let device = DeviceId(0);
    let total = footprint.device_bytes(device).unwrap();
    eprintln!(
        "== {label}: {:.2} GiB planned of {:.2} GiB",
        total as f64 / GIB,
        RTX_PRO_6000_BYTES as f64 / GIB
    );
    for (role, bytes) in footprint.role_bytes(device).unwrap() {
        eprintln!("   {role:?}: {:.2} GiB", bytes as f64 / GIB);
    }
    for owner in footprint.largest(device, 12) {
        eprintln!("   {}", owner.describe());
    }
}

/// The per-allocation report at the contexts the card names, and what dominates it.
#[test]
fn flash_next_q2_footprint_on_a_96_gb_card() {
    let (handle, _set) = handle();
    let cases = [
        ("isl 128, osl 128", 128, 128, 128),
        ("isl 1024, osl 256", 1024, 256, 1024),
        ("isl 8192, osl 256", 8192, 256, 8192),
    ];
    let mut weights = Vec::new();
    for (label, isl, osl, chunk) in cases {
        let footprint = plan(&handle, isl, osl, chunk);
        report(label, &footprint);
        let roles = footprint.role_bytes(DeviceId(0)).unwrap();
        weights.push(
            roles
                .iter()
                .find(|(role, _)| *role == BufferRole::Weight)
                .map(|&(_, bytes)| bytes)
                .unwrap(),
        );
    }
    assert!(
        weights.windows(2).all(|pair| pair[0] == pair[1]),
        "the weights do not depend on the context: {weights:?}"
    );
}

/// SC-001: the budget refuses before any allocation, typed, naming the biggest allocation and the shortfall.
/// A card that holds the plan admits it; one byte under refuses with the per-layer embedding table first.
#[test]
fn the_budget_refuses_flash_next_before_any_allocation() {
    let (handle, _set) = handle();
    let footprint = plan(&handle, 1024, 256, 1024);
    let device = DeviceId(0);
    let need = footprint.device_bytes(device).unwrap();
    let budget = |free: u64| Budget::observed(&observe(free), SUPPORT_RESERVE_BYTES);
    let fits = budget(need + SUPPORT_RESERVE_BYTES);
    assert_eq!(fits.admit(&footprint, &Ledger::default()), Ok(()));

    let short = budget(need + SUPPORT_RESERVE_BYTES - 1);
    let Err(CapacityRefusal::OverBudget {
        required,
        available,
        largest,
        ..
    }) = short.admit(&footprint, &Ledger::default())
    else {
        panic!("one byte short must be refused over budget");
    };
    assert_eq!((required, available), (need, need - 1));
    let first = &largest[0];
    assert!(
        first.id.starts_with("w.l1.ple.table"),
        "the first-named allocation is the per-layer embedding table (the weight map's `w.l1.ple.table`, GGUF `per_layer_token_embd.weight`): {}",
        first.describe()
    );
}

/// The decode-curve cell's longest context (isl 8192, osl 256) planned at each prefill chunk size: the arena
/// is set by the chunk, not by the prompt, and the largest chunk that fits the card is the one a budgeted
/// prefill would pick. Run with `--nocapture` for the table.
/// Mutation: plan every case at the whole-prompt chunk; the arena stops shrinking and the first assertion
/// fails.
#[test]
fn the_prefill_arena_is_set_by_the_chunk_not_the_prompt() {
    let (handle, _set) = handle();
    let device = DeviceId(0);
    let mut rows = Vec::new();
    for chunk in [8192usize, 4096, 2048, 1024, 512, 256, 128] {
        let footprint = plan(&handle, 8192, 256, chunk);
        let roles = footprint.role_bytes(device).unwrap();
        let of = |wanted| {
            roles
                .iter()
                .find(|(role, _)| *role == wanted)
                .map_or(0, |&(_, bytes)| bytes)
        };
        let total = footprint.device_bytes(device).unwrap();
        eprintln!(
            "chunk {chunk:>5}: weights {:.2} GiB, state {:.2} GiB, activations {:.2} GiB, total {:.2} GiB ({})",
            of(BufferRole::Weight) as f64 / GIB,
            of(BufferRole::State) as f64 / GIB,
            of(BufferRole::Activation) as f64 / GIB,
            total as f64 / GIB,
            if total + SUPPORT_RESERVE_BYTES <= RTX_PRO_6000_BYTES {
                "fits"
            } else {
                "over the card"
            }
        );
        rows.push((chunk, of(BufferRole::Activation), total));
    }
    assert!(
        rows.windows(2).all(|pair| pair[0].1 > pair[1].1),
        "a smaller chunk plans a smaller arena: {rows:?}"
    );
    assert!(
        rows.iter()
            .any(|&(_, _, total)| total + SUPPORT_RESERVE_BYTES <= RTX_PRO_6000_BYTES),
        "some chunk fits the card: {rows:?}"
    );
}

/// The chunk the benchmark runner runs a cell with on the 96 GB card, through the production choice
/// (`fit_prefill_chunk`): isl 8192 does not fit whole, so it runs at the largest candidate that does (2048:
/// 89.65 GiB planned of 95.59); isl 1024 fits whole and keeps its whole-prompt chunk; a card without room for
/// the weights is the typed refusal naming the per-layer embedding table first.
/// Mutation: make `chunk_candidates` return only the desired chunk; isl 8192 is refused instead of chunked.
#[test]
fn the_decode_curve_cell_runs_at_the_largest_chunk_that_fits_the_card() {
    let (handle, _set) = handle();
    let targets = rtx_pro_6000();
    let granule = handle.config().prefill_granule;
    let fit = |isl: usize, free: u64, candidates: &[NonZeroUsize]| {
        let (options, prompts) = cell(isl, 256, isl);
        fit_prefill_chunk(
            &handle,
            &targets,
            || observe(free),
            &options,
            &shapes(&prompts),
            candidates,
        )
    };
    let desired = |isl: usize| chunk_candidates(NonZeroUsize::new(isl).unwrap(), granule);

    let fitted = fit(8192, RTX_PRO_6000_BYTES, &desired(8192)).unwrap_or_else(|e| panic!("{e}"));
    let total = fitted
        .footprint
        .as_ref()
        .unwrap()
        .device_bytes(DeviceId(0))
        .unwrap();
    eprintln!(
        "isl 8192 on the 96 GB card: chunk {} ({:.2} GiB planned of {:.2} GiB)",
        fitted.chunk,
        total as f64 / GIB,
        RTX_PRO_6000_BYTES as f64 / GIB
    );
    assert_eq!(fitted.chunk.get(), 2048);

    let whole = fit(1024, RTX_PRO_6000_BYTES, &desired(1024)).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        whole.chunk.get(),
        1024,
        "a cell that fits whole is unchanged"
    );

    let no_room = 70 << 30;
    let candidates = &desired(8192)[..2];
    let Err(AdmitError::NoChunkFits {
        chunk,
        refusal: CapacityRefusal::OverBudget { largest, .. },
    }) = fit(8192, no_room, candidates)
    else {
        panic!("a card without room for the weights must be refused over budget");
    };
    assert_eq!(chunk.get(), 4096, "the smallest candidate tried");
    assert!(
        largest[0].id.starts_with("w.l1.ple.table"),
        "{}",
        largest[0].describe()
    );
}

/// POOT-1347: the correctness scorer's window (two heads, capacity 2112, 2048 desired) on the 96 GB card. Its
/// turns extend the prepared set to every admitted piece of both heads (`ServingShapes::turn`), which plans
/// 106.85 GiB at the whole-prompt-half chunk 1024: fitted for the prompts alone it ran at 1024, the first turn
/// then allocated the 17 GiB nobody counted and the card ran out of memory. Fitted for the turn set the chunk
/// shrinks until the whole set fits, and a card too small for the weights is the typed refusal.
/// Mutation: give `ServingShapes::turn` `Warm::Prompts(&[])`; the fit keeps 1024 and the first assertion
/// fails.
#[test]
fn the_score_window_chunk_is_fitted_to_the_turn_set() {
    let (handle, _set) = handle();
    let targets = rtx_pro_6000();
    let scored = poot_llm::Sampler::greedy().with_logprobs(20);
    let heads = [Head::serving(&scored), Head::GREEDY];
    let capacity = NonZeroUsize::new(2112).unwrap();
    let wanted = NonZeroUsize::new(2048).unwrap();
    let options = DriverOptions {
        trace: poot_llm::driver::serving_trace_limits(capacity),
        ..cell(2048, 64, 2048).0
    };
    let candidates = chunk_candidates(wanted, handle.config().prefill_granule);
    let fit = |free: u64, candidates: &[NonZeroUsize]| {
        fit_prefill_chunk(
            &handle,
            &targets,
            || observe(free),
            &options,
            &ServingShapes::turn(Layout::Contiguous, &heads),
            candidates,
        )
    };

    let fitted = fit(RTX_PRO_6000_BYTES, &candidates).unwrap_or_else(|e| panic!("{e}"));
    let planned = fitted
        .footprint
        .as_ref()
        .unwrap()
        .device_bytes(DeviceId(0))
        .unwrap();
    eprintln!(
        "score window turn set: chunk {} ({:.2} GiB planned of {:.2} GiB)",
        fitted.chunk,
        planned as f64 / GIB,
        RTX_PRO_6000_BYTES as f64 / GIB
    );
    assert!(
        fitted.chunk.get() < 1024,
        "the turn set does not fit at 1024: {}",
        fitted.chunk
    );
    assert!(
        planned + SUPPORT_RESERVE_BYTES <= RTX_PRO_6000_BYTES,
        "the fitted turn set fits the card: {planned}"
    );

    let Err(AdmitError::NoChunkFits {
        refusal:
            CapacityRefusal::OverBudget {
                required,
                available,
                ..
            },
        ..
    }) = fit(70 << 30, &candidates[..2])
    else {
        panic!("a card without room for the weights must be refused over budget");
    };
    assert!(
        required > available,
        "the shortfall is named: {required} vs {available}"
    );
}
