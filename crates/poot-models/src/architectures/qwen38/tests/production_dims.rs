//! A QSA layer at production dimensions, as `compile` leaves it (POOT-1236).
//!
//! The tracer states the block selection and the attention over it as primitive compositions
//! ([`ops::block_top_k_packed`], [`ops::selected_attention`]); the selection pass of `compile` forms
//! `OpKind::BlockTopK` and `OpKind::SelectedAttention` from them. A device runs the composites' kernels, not
//! the quadratic rank chain and the dense mask, so a program that kept the primitives would be correct and
//! unusably slow at depth: the rows below fail on the program's equations, not on its output.
//!
//! The dimensions are Flash-Next's: an indexer of 4 heads x 128 keeping 512 blocks of four, a main attention
//! of 24 query heads over 2 key heads of 256, a 128K cache (32768 blocks). The width and the other layers'
//! sizes are small: they do not enter the selection.

use poot_graph_ir::{Graph, OpKind, TraceLimits};
use poot_graph_plan::{CompileLimits, CompileOptions, Target, compile};
use poot_target::{AmdArch, Backend, DeviceCaps};

use super::*;

/// The cache capacity of a 128K context.
const CAPACITY_128K: usize = 131_072;

/// The finite limits the serving driver traces under (`poot_llm::driver::serving_trace_limits`, which this
/// crate cannot name): 2^20 values, 2^20 equations, 16 MiB of literals, 1 GiB of owned storage.
const SERVING_LIMITS: TraceLimits = TraceLimits {
    max_tokens: NonZeroUsize::MIN,
    max_values: NonZeroUsize::new(1 << 20).unwrap(),
    max_eqns: NonZeroUsize::new(1 << 20).unwrap(),
    max_literal_bytes: std::num::NonZeroU64::new(1 << 24).unwrap(),
    max_owned_bytes: std::num::NonZeroU64::new(1 << 30).unwrap(),
};

/// One linear-attention layer and one QSA layer at Flash-Next's indexer and attention dimensions.
fn production_fixture() -> Fixture {
    let mut config = fixture_config();
    let text = &mut config["text_config"];
    for (key, value) in [
        ("hidden_size", serde_json::json!(256)),
        ("num_hidden_layers", serde_json::json!(2)),
        (
            "layer_types",
            serde_json::json!(["linear_attention", "full_attention"]),
        ),
        ("num_attention_heads", serde_json::json!(24)),
        ("num_key_value_heads", serde_json::json!(2)),
        ("head_dim", serde_json::json!(256)),
        ("max_position_embeddings", serde_json::json!(CAPACITY_128K)),
        ("indexer_n_heads", serde_json::json!(4)),
        ("indexer_head_dim", serde_json::json!(128)),
        ("indexer_budget", serde_json::json!(2048)),
        ("indexer_compress_ratio", serde_json::json!(4)),
        ("ple_layer_ids", serde_json::json!([])),
    ] {
        text[key] = value;
    }
    let shapes = hf_shapes(&config, Experts::Fused);
    Fixture::zeroed_bf16(config, shapes)
}

fn target_of(backend: Backend) -> Target {
    Target::Gpu {
        backend,
        caps: match backend {
            Backend::SpirvVulkan => DeviceCaps::wgpu_rdna3_igpu(),
            Backend::Nvptx => DeviceCaps::ptx_default(),
            Backend::AmdGcn(_) => DeviceCaps::rocm_default(),
        },
    }
}

/// The decode graph of the fixture at a 128K cache, traced under the serving limits and weight-bound.
fn decode_graph(model: &dyn Model, limits: TraceLimits) -> Graph {
    let shape = StepShape::new(
        NonZeroUsize::MIN,
        NonZeroUsize::MIN,
        NonZeroUsize::new(CAPACITY_128K).unwrap(),
    )
    .with_logits(LogitRows::Last);
    let traced = model
        .trace(&StepSpec::step(shape), limits)
        .unwrap_or_else(|e| panic!("the 128K decode step traces: {e}"));
    poot_graph_plan::bind_packed_weights(
        &traced,
        &poot_graph_plan::WeightFormats::from_weight_map(model.weights()),
    )
    .unwrap()
}

/// The elements of the pairwise rank over 32768 blocks, one query: 2^30. The cache itself is 2^26.
const QUADRATIC_RANK: usize = 32_768 * 32_768;

/// The element count of the largest result of an equation of `graph`.
fn largest_value(graph: &Graph) -> usize {
    graph
        .eqns()
        .iter()
        .map(|eqn| graph.aval(eqn.out).numel())
        .max()
        .unwrap_or(0)
}

fn count(graph: &Graph, is: impl Fn(&OpKind) -> bool) -> usize {
    graph.eqns().iter().filter(|eqn| is(&eqn.op)).count()
}

/// The primitive compositions of a 128K QSA layer fit the serving trace limits: the tracer states them in a
/// few hundred equations whatever the cache holds (the extents are in the types, not in the equation count).
/// The measured size of the whole two-layer step is in the failure message of the bound, so a growth shows.
#[test]
fn the_128k_compositions_fit_the_serving_trace_limits() {
    let fx = production_fixture();
    let model = built(&fx);
    let graph = decode_graph(&*model, SERVING_LIMITS);
    let (eqns, values) = (graph.eqns().len(), graph.values().len());
    // The full-attention layer's selection and masked attention are well under 1% of the equation limit.
    assert!(
        eqns < SERVING_LIMITS.max_eqns.get() / 100
            && values < SERVING_LIMITS.max_values.get() / 100,
        "the 128K decode step holds {eqns} equations and {values} values of {} allowed",
        SERVING_LIMITS.max_eqns
    );
    assert_eq!(
        count(&graph, |op| matches!(
            op,
            OpKind::BlockTopK(_) | OpKind::SelectedAttention(_)
        )),
        0,
        "the tracer emits the compositions, not the composites"
    );
    assert!(
        largest_value(&graph) >= QUADRATIC_RANK,
        "the traced step holds the block selection's pairwise rank, {QUADRATIC_RANK} elements or more"
    );
}

/// The compiled program of the 128K QSA layer holds the composites and not the compositions, on every
/// backend: one `BlockTopK` and one `SelectedAttention` for the one QSA layer, no dense `Attention` over the
/// capacity, and no value the size of the pairwise rank over the blocks. Mutation: make the selection pass
/// decline (`form_selection` returns the graph unchanged); the rank chain has no device lowering, so `compile`
/// refuses its `Cast` (`no kernel lowers this op for these operand and output dtypes`) and this row goes red.
#[test]
fn the_128k_qsa_program_holds_the_composites_not_the_compositions() {
    let fx = production_fixture();
    let model = built(&fx);
    let graph = decode_graph(&*model, SERVING_LIMITS);
    let options = CompileOptions::new(CompileLimits::STANDARD);
    for backend in [
        Backend::AmdGcn(AmdArch::gfx1151()),
        Backend::Nvptx,
        Backend::SpirvVulkan,
    ] {
        let program = compile(&graph, &target_of(backend), &options)
            .unwrap_or_else(|e| panic!("the 128K step compiles for {backend:?}: {e}"));
        let program = program.graph();
        let of = |is: fn(&OpKind) -> bool| count(program, is);
        assert_eq!(
            of(|op| matches!(op, OpKind::BlockTopK(_))),
            1,
            "{backend:?}: the block selection is a BlockTopK"
        );
        assert_eq!(
            of(|op| matches!(op, OpKind::SelectedAttention(_))),
            1,
            "{backend:?}: the attention over it is a SelectedAttention"
        );
        assert!(
            largest_value(program) < QUADRATIC_RANK,
            "{backend:?}: an equation of {} elements remains, the pairwise rank of the 32768 blocks is {QUADRATIC_RANK}",
            largest_value(program)
        );
        assert_eq!(
            of(|op| matches!(op, OpKind::Attention(_))),
            0,
            "{backend:?}: no dense attention over the capacity remains"
        );
    }
}
