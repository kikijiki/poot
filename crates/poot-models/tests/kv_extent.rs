//! POOT-1229: stored KV extent is the planner's decision from the target's DRAM channel data. Tracers declare
//! the logical capacity; `compile` stores one position more per head where the device data says the head
//! stride camps, and exactly the capacity everywhere else.

use std::collections::HashMap;
use std::path::Path;

use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::{Graph, Slot, StateRole, Storage, ValueId};
use poot_graph_plan::{Program, compile};
use poot_models::model::{LogitRows, StepShape, StepSpec};
use poot_models::registry::Registry;
use poot_models::test_support::dense::{Dense, Family, paged_step, step};
use poot_target::dram_channels::DramChannels;
use poot_target::{AmdArch, Backend};
use poot_tensor::{DType, HostTensor};
use poot_test_util::assert_close_rel;

#[path = "support/corpus.rs"]
mod corpus;
#[path = "support/targets.rs"]
mod targets;

use corpus::{CAP, PROMPT, corpus};

/// The old padded case: 512 positions of a 128-wide head is a 2048 B head stride, a whole channel period.
const CAPACITY: usize = 512;
const HEAD_DIM: usize = 128;
const KV_HEADS: usize = 8;
const POOL_SLOTS: usize = 4 * CAPACITY;

/// Qwen3-0.6B's attention geometry: 8 KV heads of width 128.
fn dense() -> Dense {
    Dense::new(Family::Qwen3)
        .vocab(1024)
        .dims(1024, 3072, 2)
        .heads(16, KV_HEADS)
        .head_dim(HEAD_DIM)
        .max_positions(CAPACITY)
}

/// The graph `Model::trace` gives for `shape` (zero weights: only the structure is read).
fn trace(shape: StepShape) -> Graph {
    dense()
        .zeroed_model(DType::BF16)
        .model
        .trace(
            &StepSpec::step(shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .expect("the family traces a step")
}

/// The shape of every contiguous cache state of `graph` (position on axis 2).
fn cache_shapes(graph: &Graph) -> Vec<Vec<usize>> {
    graph
        .state_pairs()
        .filter(|pair| pair.role == StateRole::Positional { axis: 2 })
        .map(|pair| graph.aval(pair.input).shape.clone())
        .collect()
}

fn compiled(graph: &Graph, backend: Backend, caps: poot_target::DeviceCaps) -> Program {
    let target = poot_graph_plan::Target::Gpu { backend, caps };
    compile(graph, &target, &targets::options()).expect("the step compiles")
}

fn gfx1151() -> (Backend, poot_target::DeviceCaps) {
    let arch = AmdArch::gfx1151();
    let caps = poot_test_util::device_caps::default_caps_for(Backend::AmdGcn(arch));
    assert_eq!(caps.dram_channels, Some(DramChannels::STRIX_HALO));
    (Backend::AmdGcn(arch), caps)
}

/// A gfx part whose channel map was not measured.
fn gfx1100() -> (Backend, poot_target::DeviceCaps) {
    let arch = AmdArch::new("gfx1100", 32);
    let caps = poot_test_util::device_caps::default_caps_for(Backend::AmdGcn(arch));
    assert_eq!(caps.dram_channels, None);
    (Backend::AmdGcn(arch), caps)
}

/// SC-002: tracers see logical capacity. Every family of the corpus traces its contiguous caches at the
/// step's capacity, and the old padded case (capacity 512 of a 128-wide head) does too. Mutation: declare one
/// more position in `components::attention`'s `write_contiguous`; every row goes red.
#[test]
fn a_traced_cache_is_declared_at_the_logical_capacity() {
    let registry = Registry::builtin().expect("the builtin registry");
    let mut seen = 0;
    for row in corpus(&registry) {
        for shape in cache_shapes(&row.graph()) {
            seen += 1;
            assert!(
                [CAP, PROMPT].contains(&shape[2]),
                "{}: a cache of {} positions, over the step's capacity ({CAP} or {PROMPT})",
                row.name,
                shape[2]
            );
        }
    }
    assert!(seen > 0, "the corpus has contiguous caches");
    let graph = trace(step(1, 1, CAPACITY, LogitRows::Last));
    let shapes = cache_shapes(&graph);
    assert_eq!(shapes.len(), 4, "K and V of two layers");
    for shape in shapes {
        assert_eq!(shape, vec![1, KV_HEADS, CAPACITY, HEAD_DIM]);
    }
}

/// SC-003: planned for gfx1151 (target data with the channel period), the capacity-512 cache of a 128-wide head
/// stores a stride that does not camp, in decode, in a batched decode and in a prefill chunk; planned for a target
/// without channel data it stores exactly the capacity. Mutation: drop the `dram_channels` lookup in
/// `legalize`'s `pad_kv_extents` (never pad); the gfx1151 row goes red.
#[test]
fn a_camping_cache_is_stored_one_position_more_only_where_the_device_data_says_so() {
    let channels = DramChannels::STRIX_HALO;
    let stride = |shape: &[usize]| shape[2] * shape[3] * size_of::<f32>();
    let steps = [
        ("decode", step(1, 1, CAPACITY, LogitRows::Last)),
        ("batched decode", step(2, 1, CAPACITY, LogitRows::Last)),
        ("prefill chunk", step(1, 4, CAPACITY, LogitRows::Last)),
    ];
    for (name, shape) in steps {
        let graph = trace(shape);
        assert!(
            cache_shapes(&graph)
                .iter()
                .all(|s| channels.camps(stride(s))),
            "{name}: the traced capacity is the camping case"
        );
        let (backend, caps) = gfx1151();
        let padded = compiled(&graph, backend, caps);
        let stored = cache_shapes(padded.graph());
        assert_eq!(stored.len(), 4, "{name}");
        for s in &stored {
            assert_eq!(s[2], CAPACITY + 1, "{name}: gfx1151 stores one more");
            assert!(!channels.camps(stride(s)), "{name}: {s:?} still camps");
        }
        for (backend, caps) in [
            gfx1100(),
            (Backend::Nvptx, poot_target::DeviceCaps::ptx_default()),
        ] {
            let exact = compiled(&graph, backend, caps);
            for s in cache_shapes(exact.graph()) {
                assert_eq!(s[2], CAPACITY, "{name}: {backend:?} has no channel data");
            }
        }
    }
}

/// A paged step keeps its pool: the pool's row stride is not a head stride the planner pads.
#[test]
fn a_paged_pool_keeps_its_logical_extent() {
    let graph = trace(paged_step(1, 1, CAPACITY, POOL_SLOTS, LogitRows::Last));
    let (backend, caps) = gfx1151();
    let program = compiled(&graph, backend, caps);
    for pair in program.graph().state_pairs() {
        assert_eq!(
            program.graph().aval(pair.input).shape,
            vec![POOL_SLOTS, KV_HEADS, HEAD_DIM]
        );
    }
}

/// Deterministic pseudo-random fill in [-1, 1) seeded by `seed` (the helper the other pass tests carry).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// A stable seed from a value's name, so two graphs of one model bind the same data to the same value.
fn seed_of(name: &str) -> u64 {
    name.bytes()
        .fold(7, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)))
}

/// One step of `tokens` consecutive tokens from `pos` of the graph over its inputs: weights and the logical caches
/// filled from the value's name, the caches widened to `graph`'s stored extent with `pad` in every head's extra
/// positions.
fn step_logits(graph: &Graph, logical_cap: usize, tokens: usize, pos: i32, pad: f32) -> Vec<f32> {
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in graph.inputs() {
        let meta = graph.meta(id);
        let aval = meta.aval.clone();
        let tensor = match meta.storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(aval.shape, vec![5; tokens]),
            Storage::Slot(Slot::Pos) => {
                HostTensor::i32(aval.shape, (pos..pos + tokens as i32).collect())
            }
            Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
            Storage::Computed(c) => c.tensor(),
            Storage::Const | Storage::State => {
                let name = meta.name().expect("a bound input is named").into_owned();
                let stored =
                    aval.shape.get(2).copied().filter(|_| {
                        graph.state_role(id) == Some(StateRole::Positional { axis: 2 })
                    });
                let data: Vec<f32> = match stored {
                    Some(stored) => {
                        let (heads, d) = (aval.shape[1], aval.shape[3]);
                        let logical = fill(heads * logical_cap * d, seed_of(&name));
                        logical
                            .chunks(logical_cap * d)
                            .flat_map(|head| {
                                head.iter()
                                    .copied()
                                    .chain(std::iter::repeat_n(pad, (stored - logical_cap) * d))
                            })
                            .collect()
                    }
                    None => fill(aval.numel(), seed_of(&name))
                        .iter()
                        .map(|v| v * 0.1)
                        .collect(),
                };
                HostTensor::f32(aval.shape, data)
            }
            Storage::Device => unreachable!("a device value is not an input"),
        };
        inputs.insert(id, Value::from(tensor));
    }
    eval(graph, &inputs, EvalOptions::default())
        .expect("the step evaluates")
        .output
        .into_host()
        .expect("a host output")
        .as_f32()
        .expect("f32 logits")
        .to_vec()
}

/// SC-005 (model-free half): the stored extra position never reaches the logits, even when it holds a large finite
/// value. A decode step bounds its attention by the live length; a prefill chunk reads the whole stored extent
/// under the mask, which hides the extra column. Mutation: hide the pad column with `0.0` instead of
/// `MASK_HIDDEN` in `legalize`'s `hide_pad_column`; the prefill row weighs the extra position and goes red.
#[test]
fn the_padded_position_never_reaches_the_logits() {
    let cap = 64;
    let model = Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, 2)
        .heads(4, 2)
        .head_dim(8)
        .max_positions(cap)
        .f32_model();
    for (name, tokens) in [("decode", 1), ("prefill chunk", 4)] {
        let graph = model
            .model
            .trace(
                &StepSpec::step(step(1, tokens, cap, LogitRows::Last)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .expect("the family traces a step");
        let (backend, caps) = gfx1151();
        let program = compiled(&graph, backend, caps);
        for s in cache_shapes(program.graph()) {
            assert_eq!(
                s[2],
                cap + 1,
                "{name}: the fixture capacity camps and is padded"
            );
        }
        let want = step_logits(&graph, cap, tokens, 5, 0.0);
        for pad in [0.0, 1.0e4] {
            let got = step_logits(program.graph(), cap, tokens, 5, pad);
            assert_close_rel(&got, &want, 1e-5);
        }
    }
}

/// Every `.rs` file under `dir`.
fn sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("a readable source directory") {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// SC-001: no model code names a DRAM channel rule. The layout rule is the planner's, from the target's data
/// (`channel` alone is a conv or SSM channel). Mutation: restore `components/kv_stride.rs`; its DRAM
/// vocabulary is found and this goes red.
#[test]
fn no_model_code_names_the_dram_channel_rule() {
    // `std::env::var`, not `env!`: see plan_summary.rs (a shared compile cache must not bake a worktree path).
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let mut files = Vec::new();
    sources(&Path::new(&manifest).join("src"), &mut files);
    assert!(files.len() > 50, "the model sources were found");
    let forbidden = [
        "dram",
        "camping",
        "camps",
        "channel period",
        "channel data",
        "channel map",
    ];
    let mut found = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(&file)
            .expect("a UTF-8 source")
            .to_lowercase();
        for word in forbidden {
            if text.contains(word) {
                found.push(format!("{} names {word:?}", file.display()));
            }
        }
    }
    assert!(found.is_empty(), "{}", found.join("\n"));
}
