//! POOT-1139: the planner's stride audit. A float buffer whose row stride has `gcd(stride / 256, 16) > 4`
//! camps on at most two of the APU's sixteen DRAM channels (`poot_test_util::dram_channels::alias_factor`):
//! the strided-read probe in the device crates measures what that costs. This test compiles every plan-summary
//! corpus graph and two real decode configurations for the three reference targets and writes one line per
//! (buffer, axis) of an F16 or F32 buffer that has such a stride on a non-innermost axis with extent above one.
//! The lines are compared with the checked-in `tests/stride_audit.tsv`, so a layout change that adds, removes
//! or moves an aliased buffer names it. The file is sorted by buffer bytes, largest first, so the real KV caches
//! lead it. It is an audit: no layout changes here, the file is the list a follow-up
//! card decides from.
//!
//! What a buffer's strides are: the planner gives every graph input and every computed value (an equation that
//! dispatches) a dense row-major buffer, so axis `a` has the stride `product(shape[a+1..]) * width`. Aliases and
//! strided views own no buffer; a view reads its source's strides and is audited through the source. Packed
//! and quantized storage is out of scope (a packed weight row is a whole number of words and is not a float
//! row), as are BF16 buffers and per-equation scratch (one-dimensional).

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::PathBuf;

use poot_graph_ir::op::BinOp;
use poot_graph_ir::{Builder, Graph, Outputs, Slot, StateRole, Storage, TensorType, ValueId};
use poot_graph_plan::{Plan, Program, compile};
use poot_models::model::{LogitRows, Phase};
use poot_models::registry::Registry;
use poot_models::test_support::dense::{Dense, Family, step};
use poot_target::storage::{LogicalDType, StorageLayout};
use poot_tensor::DType;
use poot_test_util::dram_channels::alias_factor;

#[path = "support/corpus.rs"]
mod corpus;
#[path = "support/targets.rs"]
mod targets;

use corpus::corpus;
use targets::{RefTarget, reference_targets};

/// Rewrites the checked-in file instead of comparing with it.
const REGEN_ENV: &str = "POOT_STRIDE_AUDIT_REGEN";

/// A stride is reported when its alias factor is above this: more than four of the sixteen channels idle.
const ALIAS_FACTOR_ABOVE: usize = 4;

const HEADER: &str =
    "graph\ttarget\tbuffer\trole\tdtype\tshape\taxis\tstride_bytes\talias\tbytes\n";

/// One float buffer axis whose stride camps on few channels.
#[derive(Debug, PartialEq, Eq)]
struct Finding {
    buffer: String,
    role: &'static str,
    dtype: DType,
    shape: Vec<usize>,
    axis: usize,
    stride_bytes: usize,
    alias: usize,
    bytes: usize,
}

/// How a graph value is named in the audit: its bound name for an input, `eqn<i>:<op>` for a computed value.
fn buffer_name(graph: &Graph, id: ValueId) -> String {
    match graph.eqns().iter().position(|eqn| eqn.out == id) {
        Some(index) => format!("eqn{index}:{}", graph.eqns()[index].op.name()),
        None => graph
            .meta(id)
            .name
            .clone()
            .unwrap_or_else(|| format!("{id}")),
    }
}

fn role(storage: Storage) -> &'static str {
    match storage {
        Storage::Const => "weight",
        Storage::Computed(_) => "computed_const",
        Storage::Slot(_) => "slot",
        Storage::State => "state",
        Storage::Device => "computed",
    }
}

/// Every aliased float buffer axis of `program`'s compiled graph.
fn audit(program: &Program) -> Vec<Finding> {
    let graph = program.graph();
    // A value another value merely views owns no buffer of its own.
    let viewed: HashSet<ValueId> = program
        .planned()
        .filter(|(_, plan)| !matches!(plan, Plan::Dispatches(_)))
        .map(|(eqn, _)| eqn.out)
        .collect();
    // The value table also keeps the ids optimization removed; a buffer exists for an input or an equation's output.
    let live: HashSet<ValueId> = graph
        .inputs()
        .iter()
        .copied()
        .chain(graph.eqns().iter().map(|eqn| eqn.out))
        .collect();
    let mut findings = Vec::new();
    for id in graph.values().ids() {
        if viewed.contains(&id) || !live.contains(&id) {
            continue;
        }
        let meta = graph.meta(id);
        let storage = program.storage().storage(id).buffer_storage();
        let float = matches!(storage.dtype(), LogicalDType::F16 | LogicalDType::F32);
        if storage.layout() != StorageLayout::Dense || !float {
            continue;
        }
        let shape = &meta.aval.shape;
        let width = storage.element().byte_width();
        let bytes = shape.iter().product::<usize>() * width;
        for axis in 0..shape.len().saturating_sub(1) {
            if shape[axis] <= 1 {
                continue;
            }
            let stride_bytes = shape[axis + 1..].iter().product::<usize>() * width;
            let alias = alias_factor(stride_bytes);
            if alias > ALIAS_FACTOR_ABOVE {
                findings.push(Finding {
                    buffer: buffer_name(graph, id),
                    role: role(meta.storage),
                    dtype: meta.aval.dtype,
                    shape: shape.clone(),
                    axis,
                    stride_bytes,
                    alias,
                    bytes,
                });
            }
        }
    }
    findings
}

// ---- the audited graphs -----------------------------------------------------------------------

/// Qwen2.5-0.5B's dimensions: 2 KV heads of width 64.
fn qwen2_0_5b() -> Dense {
    Dense::new(Family::Qwen2)
        .vocab(151936)
        .dims(896, 4864, 24)
        .heads(14, 2)
        .head_dim(64)
        .max_positions(32768)
}

/// Qwen3-0.6B's dimensions: 8 KV heads of width 128.
fn qwen3_0_6b() -> Dense {
    Dense::new(Family::Qwen3)
        .vocab(151936)
        .dims(1024, 3072, 28)
        .heads(16, 8)
        .head_dim(128)
        .max_positions(40960)
}

/// KV capacity of the real decode graphs: a 2048-position cache, a power of two as serving configures.
const REAL_CAP: usize = 2048;

/// The graph `Model::trace` gives for one decode token of `dense` (zero weights: only the structure is read).
fn real_decode(dense: &Dense) -> Graph {
    dense
        .zeroed_model(DType::BF16)
        .model
        .trace(
            Phase::Decode,
            step(1, 1, REAL_CAP, LogitRows::Last),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .expect("the family traces a decode step")
}

/// Every audited graph with its name: the plan-summary corpus, then the real decode configs.
fn graphs() -> Vec<(String, Graph)> {
    let registry = Registry::builtin().expect("the builtin registry");
    let mut graphs: Vec<(String, Graph)> = corpus(&registry)
        .into_iter()
        .map(|row| (row.name.clone(), row.graph()))
        .collect();
    graphs.push((
        format!("real.qwen2_0_5b.decode.cap{REAL_CAP}"),
        real_decode(&qwen2_0_5b()),
    ));
    graphs.push((
        format!("real.qwen3_0_6b.decode.cap{REAL_CAP}"),
        real_decode(&qwen3_0_6b()),
    ));
    graphs
}

fn render() -> String {
    let options = targets::options();
    let graphs = graphs();
    let mut rows: Vec<(usize, String)> = Vec::new();
    for RefTarget { file, target, .. } in reference_targets() {
        for (name, graph) in &graphs {
            // A refused graph has no plan and so no layout; plan_summary pins the refusal.
            let Ok(program) = compile(graph, &target, &options) else {
                continue;
            };
            for f in audit(&program) {
                let line = format!(
                    "{name}\t{file}\t{}\t{}\t{:?}\t{:?}\t{}\t{}\t{}\t{}",
                    f.buffer, f.role, f.dtype, f.shape, f.axis, f.stride_bytes, f.alias, f.bytes,
                );
                rows.push((f.bytes, line));
            }
        }
    }
    // Largest buffers first (a stable sort: ties keep target, graph and value order), so the real KV caches
    // lead the file and the fixture-sized buffers trail it.
    rows.sort_by_key(|&(bytes, _)| std::cmp::Reverse(bytes));
    let mut out = String::from(HEADER);
    for (_, line) in rows {
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn expected_path() -> PathBuf {
    // `std::env::var`, not `env!`: see plan_summary.rs (a shared compile cache must not bake a worktree path).
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set by cargo for test binaries");
    PathBuf::from(manifest_dir)
        .join("tests")
        .join("stride_audit.tsv")
}

/// Lines shown in a failure before the rest are counted.
const DIFF_LIMIT: usize = 24;

/// The lines in `to` and not in `from`, each named by its graph, target and buffer, then the stride columns.
fn difference(label: &str, from: &str, to: &str) -> String {
    let have: HashSet<&str> = from.lines().collect();
    let lines: Vec<String> = to
        .lines()
        .filter(|line| !have.contains(line))
        .map(|line| {
            let c: Vec<&str> = line.split('\t').collect();
            format!(
                "  {label} {} [{}] {}: {} {} shape {} axis {} stride {} B alias {} ({} B)",
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]
            )
        })
        .collect();
    let mut out = lines[..lines.len().min(DIFF_LIMIT)].join("\n");
    if lines.len() > DIFF_LIMIT {
        write!(out, "\n  ... and {} more {label}", lines.len() - DIFF_LIMIT)
            .expect("write to a String");
    }
    out
}

/// The aliased float buffers over the corpus and the real decode configs are exactly the checked-in list.
/// A buffer whose row stride becomes (or stops being) a multiple of 2048 B shows as an added (removed) line
/// naming the graph, target and buffer. Mutation: pad one KV row stride to a multiple of 4096 B (the
/// `head_dim` of a corpus family's KV cache, or `Qwen2.5-0.5B`'s `head_dim(64)` above to `1024`); the row for
/// that cache goes red as `added`.
#[test]
fn the_aliased_float_buffers_are_the_checked_in_list() {
    let actual = render();
    let path = expected_path();
    if std::env::var_os(REGEN_ENV).is_some() {
        std::fs::write(&path, &actual).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e}\nregenerate with `{REGEN_ENV}=1 cargo nextest run -p poot-models --test stride_audit`",
            path.display()
        )
    });
    if expected != actual {
        panic!(
            "the aliased float buffers differ from {}\n{}\n{}\n\
             If the layout change is intended, regenerate with `{REGEN_ENV}=1 cargo nextest run -p poot-models \
             --test stride_audit` and list the changed buffers in the card's landing note.",
            path.display(),
            difference("added", &expected, &actual),
            difference("removed", &actual, &expected),
        );
    }
}

/// The audit is not vacuous: over a graph with one state buffer whose head stride is 4096 B and one whose
/// stride is 4160 B (a 64 B pad per row), it names the first and not the second, with the stride, alias
/// factor and size. Mutation: have `audit` skip `StorageLayout::Dense` buffers, or compare with `>= 16`.
#[test]
fn the_audit_names_a_four_kib_stride_and_not_its_padded_twin() {
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![4, 1024]))
        .unwrap();
    let aliased = b
        .state_input(
            "kv_aliased",
            TensorType::f32(vec![4, 1024]),
            StateRole::Recurrent,
        )
        .unwrap();
    let padded = b
        .state_input(
            "kv_padded",
            TensorType::f32(vec![4, 1040]),
            StateRole::Recurrent,
        )
        .unwrap();
    let next_aliased = b.binary(BinOp::Add, aliased, x).unwrap();
    let head = b.slice(padded, 1, 0, 1024).unwrap();
    let next_padded_head = b.binary(BinOp::Add, head, x).unwrap();
    let tail = b.slice(padded, 1, 1024, 1040).unwrap();
    let next_padded = b.concat(1, &[next_padded_head, tail]).unwrap();
    let graph = b
        .finish(
            Outputs::new(next_aliased)
                .with_state(&[(aliased, next_aliased), (padded, next_padded)]),
        )
        .unwrap();
    let target = targets::preset("wgpu", "spirv_vulkan", poot_target::Backend::SpirvVulkan).target;
    let program = compile(&graph, &target, &targets::options()).expect("the graph compiles");
    let found = audit(&program);
    let named = |name: &str| found.iter().any(|f| f.buffer == name);
    assert!(
        named("kv_aliased"),
        "the 4096 B stride is not named: {found:?}"
    );
    assert!(!named("kv_padded"), "the 4160 B stride is named: {found:?}");
    let hit = found.iter().find(|f| f.buffer == "kv_aliased").unwrap();
    assert_eq!(
        (hit.axis, hit.stride_bytes, hit.alias, hit.bytes),
        (0, 4096, 16, 4 * 4096)
    );
}
