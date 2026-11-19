//! Attention fixtures: flash decode and prefill under a per-head ALiBi mask, and decode over the
//! query/key-value head layouts (multi-head, multi-query) the shared qwen2 decode fixture does not
//! cover.

use poot_eval::test_support::StepFixture;
use poot_executor::Executor;
use poot_graph_ir::ops::{attention_masked, attention_prefill_softcap};
use poot_graph_ir::{Builder, TensorType};
use poot_tensor::HostTensor;

use crate::{Fixture, const_fixture};

const HQ: usize = 4;
const HKV: usize = 2;
const D: usize = 4;

/// One ALiBi case: the same attention over the same q/k/v, once with the per-head mask and once
/// with head 0's mask row broadcast to every head.
pub struct AlibiCase {
    /// The mask holds a different slope for every head.
    pub per_head: Fixture,
    /// The same program with a broadcast mask (`Hm = 1`) bound to head 0's row. A device that read
    /// the per-head mask at head stride 0 would produce this output, so the oracle's output for
    /// `per_head` must differ from this fixture's: a row asserts it, which shows `per_head` can tell
    /// the two apart.
    pub head0_broadcast: Fixture,
    /// The prefix of the planned op name this case runs as (`attention`).
    pub flash_op: &'static str,
}

/// Per-head ALiBi slopes: a distinct power of two for each head.
const SLOPES: [f32; HQ] = [0.25, 0.0625, 0.015625, 0.00390625];

fn centered(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// The three attention shapes the generated kernel serves: a decode, a prefill, and a softcapped
/// prefill. The per-head mask is a real ALiBi mask: visibility plus `-slope[h] * (query - key)`.
pub fn alibi_flash_cases() -> Vec<AlibiCase> {
    vec![
        alibi_decode_case(),
        alibi_prefill_case("alibi_flash_prefill_region", None),
        alibi_prefill_case("alibi_flash_prefill_imported", Some(50.0)),
    ]
}

fn alibi_decode_case() -> AlibiCase {
    let (cap, pos) = (6usize, 3usize);
    let mask: Vec<f32> = (0..HQ)
        .flat_map(|h| {
            (0..cap).map(move |t| {
                if t <= pos {
                    -SLOPES[h] * (pos as f32 - t as f32)
                } else {
                    -1.0e9
                }
            })
        })
        .collect();
    let build = |name: &'static str, mask_heads: usize, mask: Vec<f32>| {
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, HQ, 1, D])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, HKV, cap, D]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, HKV, cap, D]))
            .unwrap();
        let m = b
            .constant("mask", TensorType::f32(vec![1, mask_heads, 1, cap]))
            .unwrap();
        let y = attention_masked(&b, q, k, v, HQ / HKV, 1.0 / (D as f32).sqrt(), m).unwrap();
        let graph = b.finish(y).unwrap();
        const_fixture(
            name,
            graph,
            vec![
                ("q", HostTensor::f32(vec![1, HQ, 1, D], centered(1, HQ * D))),
                (
                    "k",
                    HostTensor::f32(vec![1, HKV, cap, D], centered(2, HKV * cap * D)),
                ),
                (
                    "v",
                    HostTensor::f32(vec![1, HKV, cap, D], centered(3, HKV * cap * D)),
                ),
                ("mask", HostTensor::f32(vec![1, mask_heads, 1, cap], mask)),
            ],
            2,
        )
    };
    AlibiCase {
        head0_broadcast: build("alibi_flash_decode_head0", 1, mask[..cap].to_vec()),
        per_head: build("alibi_flash_decode", HQ, mask),
        flash_op: "attention",
    }
}

fn alibi_prefill_case(name: &'static str, softcap: Option<f32>) -> AlibiCase {
    let l = 6usize;
    let mask: Vec<f32> = (0..HQ)
        .flat_map(|h| {
            (0..l).flat_map(move |i| {
                (0..l).map(move |j| {
                    if j <= i {
                        -SLOPES[h] * (i as f32 - j as f32)
                    } else {
                        -1.0e30
                    }
                })
            })
        })
        .collect();
    let build = |name: &'static str, mask_heads: usize, mask: Vec<f32>| {
        let b = Builder::for_tests();
        let q = b.constant("q", TensorType::f32(vec![1, HQ, l, D])).unwrap();
        let k = b
            .constant("k", TensorType::f32(vec![1, HKV, l, D]))
            .unwrap();
        let v = b
            .constant("v", TensorType::f32(vec![1, HKV, l, D]))
            .unwrap();
        let m = b
            .constant("m", TensorType::f32(vec![1, mask_heads, l, l]))
            .unwrap();
        let y =
            attention_prefill_softcap(&b, q, k, v, HQ / HKV, 1.0 / (D as f32).sqrt(), m, softcap)
                .unwrap();
        let graph = b.finish(y).unwrap();
        const_fixture(
            name,
            graph,
            vec![
                (
                    "q",
                    HostTensor::f32(vec![1, HQ, l, D], centered(4, HQ * l * D)),
                ),
                (
                    "k",
                    HostTensor::f32(vec![1, HKV, l, D], centered(5, HKV * l * D)),
                ),
                (
                    "v",
                    HostTensor::f32(vec![1, HKV, l, D], centered(6, HKV * l * D)),
                ),
                ("m", HostTensor::f32(vec![1, mask_heads, l, l], mask)),
            ],
            2,
        )
    };
    AlibiCase {
        head0_broadcast: build("alibi_flash_prefill_head0", 1, mask[..l * l].to_vec()),
        per_head: build(name, HQ, mask),
        flash_op: "attention",
    }
}

// ---- Paged attention reads the KV pools in place ----

/// Batch rows, pool blocks and positions of the paged fixture. The pool holds more blocks than the rows
/// together read, so a block table that addressed the pool by position would read the wrong rows.
const PAGED_BATCH: usize = 2;
const PAGED_BLOCK: usize = 4;
const PAGED_BLOCKS: usize = 4;
const PAGED_SLOTS: usize = PAGED_BLOCKS * PAGED_BLOCK;
const PAGED_CAP: usize = 8;

/// A GQA decode (four query heads over two key heads, two batch rows) over the `[16, 2, 4]` K and V pools of
/// a paged cache, read through a `[2, 2]` block table of four-slot blocks ([`Builder::block_table_slot`]) the
/// way the models trace it. Two steps bind two different scattered tables over the same pools, so the second
/// is a replay that must follow the new table, not the one the recording read. Row 1 sees all eight positions,
/// row 0 only four; the blocks past a row's length name block 0, as the engine's table does for a block not
/// written yet.
/// The program must plan one attention equation over the pools themselves: no gather, no transpose.
pub fn paged_attention_fixture() -> Fixture {
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![PAGED_BATCH, HQ, 1, D]))
        .unwrap();
    let k_pool = b
        .constant("k_pool", TensorType::f32(vec![PAGED_SLOTS, HKV, D]))
        .unwrap();
    let v_pool = b
        .constant("v_pool", TensorType::f32(vec![PAGED_SLOTS, HKV, D]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![PAGED_BATCH, 1, 1, PAGED_CAP]))
        .unwrap();
    let read = b
        .block_table_slot("read", PAGED_BATCH, PAGED_CAP / PAGED_BLOCK, PAGED_BLOCKS)
        .unwrap();
    let block = std::num::NonZeroUsize::new(PAGED_BLOCK).unwrap();
    let k = poot_graph_ir::ops::paged_view(&b, k_pool, read, block).unwrap();
    let v = poot_graph_ir::ops::paged_view(&b, v_pool, read, block).unwrap();
    let y = attention_masked(&b, q, k, v, HQ / HKV, 1.0 / (D as f32).sqrt(), mask).unwrap();
    let read = read.id();
    let graph = b.finish(y).unwrap();
    let key = graph.meta(read).slot_key().unwrap().clone();
    let mut visible = vec![0.0f32; PAGED_BATCH * PAGED_CAP];
    visible[4..PAGED_CAP].fill(-1.0e9);
    let mut fixture = const_fixture(
        "paged_attention",
        graph,
        vec![
            (
                "q",
                HostTensor::f32(
                    vec![PAGED_BATCH, HQ, 1, D],
                    centered(1, PAGED_BATCH * HQ * D),
                ),
            ),
            (
                "k_pool",
                HostTensor::f32(
                    vec![PAGED_SLOTS, HKV, D],
                    centered(2, PAGED_SLOTS * HKV * D),
                ),
            ),
            (
                "v_pool",
                HostTensor::f32(
                    vec![PAGED_SLOTS, HKV, D],
                    centered(3, PAGED_SLOTS * HKV * D),
                ),
            ),
            (
                "mask",
                HostTensor::f32(vec![PAGED_BATCH, 1, 1, PAGED_CAP], visible),
            ),
        ],
        0,
    );
    let maps: [[i32; PAGED_BATCH * PAGED_CAP / PAGED_BLOCK]; 2] = [[3, 0, 1, 2], [2, 0, 3, 1]];
    fixture.steps = maps
        .iter()
        .map(|words| {
            vec![StepFixture {
                key: key.clone(),
                tensor: HostTensor::i32(vec![PAGED_BATCH, PAGED_CAP / PAGED_BLOCK], words.to_vec()),
            }]
        })
        .collect();
    fixture
}

/// [`paged_attention_fixture`] on `exec` at `target`: the program must plan the attention equation over the
/// pools themselves (no gather, no transpose of a materialized view), and each of its two steps must match the
/// CPU oracle's evaluation of the materialized gathered view, within the parity tolerance.
pub fn assert_paged_attention_matches_the_oracle(
    exec: &mut dyn Executor,
    target: poot_graph_plan::Target,
) -> Result<(), String> {
    let fixture = paged_attention_fixture();
    let ops = crate::planned_ops(&fixture, target);
    if !ops.iter().any(|op| op.contains("key=Paged")) {
        return Err(format!(
            "paged attention: no attention reads the pools through the slot map: {ops:?}"
        ));
    }
    if let Some(movement) = ops
        .iter()
        .find(|op| op.starts_with("gather") || op.starts_with("transpose"))
    {
        return Err(format!(
            "paged attention: the plan still materializes the gathered view ({movement}): {ops:?}"
        ));
    }
    let mut oracle =
        poot_eval::test_support::weight_map_oracle::oracle_for(&fixture.graph, &fixture.store);
    crate::run_parity(exec, target, &fixture, &mut oracle)
}

// ---- Split-KV decode attention (POOT-1113) ----

/// Two query rows against a 1024-key cache of one key/value head: the low-head decode a 40-compute-unit
/// device leaves almost idle, which the planner gives a grid of eight splits (of at least 128 keys each).
fn split_kv_fixture() -> poot_eval::test_support::live_attention::LiveAttentionFixture {
    poot_eval::test_support::live_attention::LiveAttentionFixture {
        batch: 1,
        hq: 2,
        hkv: 1,
        cap: 1024,
        dk: 64,
        dv: 64,
        shared_dim: None,
        softcap: None,
        dtype: poot_tensor::DType::F32,
    }
}

/// The live length of each step of the split-KV rows, in order. The first fills the cache, so all eight splits
/// publish real partials into the arena's scratch slot; the second needs one share of the eight, so a combine
/// that read past the splits its row published would take the first step's stale partials; then a mid-length
/// that ends inside a share, a one-key length, and the full cache again. Each step is compared against an
/// independent evaluation of that length.
pub const SPLIT_KV_LIVE_LENGTHS: [i32; 5] = [1024, 5, 400, 1, 1024];

/// How many dispatches and scratch buffers the live decode's attention plans at `target`: `(2, 1)` for the
/// split plan, `(1, 0)` for streaming. A row asserts it so a device row cannot pass by the planner quietly
/// declining the split on the device's own caps. (The executor refuses a program compiled for caps other
/// than its device's, so a streaming plan of the same fixture cannot be run on the device: the
/// kernel-level rows in `poot-kernelgen` compare the two schedules, and the device rows compare the split
/// plan with an independent evaluation.)
pub fn attention_launches(
    built: &poot_eval::test_support::live_attention::LiveAttentionGraph,
    target: poot_graph_plan::Target,
) -> (usize, usize) {
    let program = crate::live_attention::compile_live(&built.graph, target);
    let mut found = None;
    for (_, _, stage) in program.stages() {
        if let Some(launches) = stage_attention_launches(stage) {
            assert!(found.is_none(), "exactly one attention equation");
            found = Some(launches);
        }
    }
    found.expect("the live decode keeps one attention equation")
}

/// The `(dispatches, scratch buffers)` of the one attention equation of `stage`, or `None` when it has none.
fn stage_attention_launches(stage: &poot_graph_plan::Program) -> Option<(usize, usize)> {
    let mut found = None;
    for (eqn, plan) in stage.planned() {
        if matches!(eqn.op, poot_graph_ir::OpKind::Attention(_)) {
            assert!(found.is_none(), "at most one attention equation per stage");
            let scratch = match plan {
                poot_graph_plan::Plan::Dispatches(dispatches) => dispatches.scratch().len(),
                other => panic!("an attention launches: {other:?}"),
            };
            found = Some((plan.dispatches().len(), scratch));
        }
    }
    found
}

/// The absolute tolerance of a split-KV device row against the CPU oracle's `f32` evaluation. The split
/// reassociates the key sum (per-split partials, then the cross-split combine) on top of the streaming
/// kernel's own per-lane reassociation, and a device `exp` is not the oracle's; the operands are uniform in
/// `[-1, 1)` and the output is a softmax-weighted mean of value rows, so it is `O(0.1)`. `1e-4` is two orders
/// of magnitude above the error the interpreter rows observe between the two schedules and far below what
/// a dropped split moves the output by (`O(1/splits)` of its magnitude).
pub const SPLIT_KV_DEVICE_TOLERANCE: f32 = 1e-4;

fn split_kv_outputs(
    exec: &mut dyn Executor,
    target: poot_graph_plan::Target,
    passes: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let fixture = split_kv_fixture();
    let built = fixture.build(true);
    let planned = attention_launches(&built, target);
    if planned != (2, 1) {
        return Err(format!(
            "the live decode planned (dispatches, scratch) = {planned:?}, not the split plan (2, 1)"
        ));
    }
    let inputs = fixture.inputs(&built, &[SPLIT_KV_LIVE_LENGTHS[0]], false);
    let positions: Vec<Vec<i32>> = (0..passes)
        .flat_map(|_| SPLIT_KV_LIVE_LENGTHS.iter().map(|&len| vec![len - 1]))
        .collect();
    crate::live_attention::live_steps(exec, target, &built, &inputs, &positions)
        .map_err(|e| format!("split-KV run: {e}"))
}

/// Run the low-head live decode at [`SPLIT_KV_LIVE_LENGTHS`] over one executable on `exec` at `target`:
/// the plan must be the split plan, and every step's output must match the CPU oracle's evaluation of that
/// live length within [`SPLIT_KV_DEVICE_TOLERANCE`]. The scratch slot is stale from the previous step at
/// every step but the first.
pub fn assert_split_kv_decode_matches_the_oracle(
    exec: &mut dyn Executor,
    target: poot_graph_plan::Target,
) -> Result<(), String> {
    let split = split_kv_outputs(exec, target, 1)?;
    let fixture = split_kv_fixture();
    let built = fixture.build(true);
    for (i, &len) in SPLIT_KV_LIVE_LENGTHS.iter().enumerate() {
        let inputs = fixture.inputs(&built, &[len], false);
        let want = fixture.oracle(&inputs, &built, &[len]);
        if split[i].len() != want.len() {
            return Err(format!(
                "split-KV live length {len}: {} elements against the oracle's {}",
                split[i].len(),
                want.len()
            ));
        }
        // `max_abs_error` panics on a NaN or an infinity, naming the element: a non-finite output fails
        // the row at any tolerance.
        let worst = poot_test_util::max_abs_error(&split[i], &want);
        if worst > SPLIT_KV_DEVICE_TOLERANCE {
            return Err(format!(
                "split-KV live length {len}: worst absolute error {worst} against the oracle exceeds {SPLIT_KV_DEVICE_TOLERANCE}"
            ));
        }
    }
    Ok(())
}

/// The same entry stepped through [`SPLIT_KV_LIVE_LENGTHS`] twice: the second pass is every step a pure
/// replay of the captured two-launch plan, and its outputs must equal the first pass's bit for bit (the
/// first pass's first step is the recording step's own execution). A replay that dropped the combine, or
/// ran the launches out of order, differs.
pub fn assert_split_kv_decode_replays_bit_identically(
    exec: &mut dyn Executor,
    target: poot_graph_plan::Target,
) -> Result<(), String> {
    let split = split_kv_outputs(exec, target, 2)?;
    let (first, second) = split.split_at(SPLIT_KV_LIVE_LENGTHS.len());
    for (i, (a, b)) in first.iter().zip(second).enumerate() {
        let same = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        if !same {
            return Err(format!(
                "split-KV replay of live length {} is not bit-identical to the first pass",
                SPLIT_KV_LIVE_LENGTHS[i]
            ));
        }
    }
    Ok(())
}

/// The live prefix lengths of [`assert_split_kv_decode_is_capacity_independent`]: inside one share, at and just
/// past the planner's 128-key floor (two shares, the second a single key), a ragged second share, and the
/// whole small cache.
const CAPACITY_LIVE_LENGTHS: [i32; 5] = [5, 128, 129, 200, 256];
/// The capacities of [`assert_split_kv_decode_is_capacity_independent`].
const SMALL_CAPACITY: usize = 256;
const LARGE_CAPACITY: usize = 4096;

/// The splits of the one attention equation of `built` compiled for `target`.
fn planned_grid(
    built: &poot_eval::test_support::live_attention::LiveAttentionGraph,
    target: poot_graph_plan::Target,
) -> Result<usize, String> {
    let program = crate::live_attention::compile_live(&built.graph, target);
    let mut found = None;
    for (_, _, stage) in program.stages() {
        for (eqn, _) in stage.planned() {
            if !matches!(eqn.op, poot_graph_ir::OpKind::Attention(_)) {
                continue;
            }
            let poot_graph_plan::KernelChoice::Generated(poot_kernelgen::KernelRequest::Attention(
                spec,
            )) = stage.kernel_choice(eqn)
            else {
                return Err("the live decode did not plan a generated attention".to_string());
            };
            let poot_kernelgen::AttentionSchedule::SplitKv { splits, .. } = spec.schedule else {
                return Err(format!(
                    "the live decode did not plan the split plan: {spec:?}"
                ));
            };
            found = Some(splits);
        }
    }
    found.ok_or_else(|| "the live decode keeps one attention equation".to_string())
}

/// The split-KV decode over the same live prefix of a 256-key and of a 4096-key cache gives bit-identical
/// outputs (POOT-1400). The two plan different grids (the capacity bounds the grid), but the live length alone
/// partitions it, and a prefix of at most 256 keys has at most two shares in both. The keys past the live
/// length are NaN in both caches, so a share that strayed past the length poisons its output, and the prefix
/// holds the same values in both. One step per live length in [`CAPACITY_LIVE_LENGTHS`], each through the
/// executor's recording step.
pub fn assert_split_kv_decode_is_capacity_independent(
    exec: &mut dyn Executor,
    target: poot_graph_plan::Target,
) -> Result<(), String> {
    let at = |cap| poot_eval::test_support::live_attention::LiveAttentionFixture {
        cap,
        ..split_kv_fixture()
    };
    let (small, large) = (at(SMALL_CAPACITY), at(LARGE_CAPACITY));
    let (small_built, large_built) = (small.build(true), large.build(true));
    let (small_grid, large_grid) = (
        planned_grid(&small_built, target)?,
        planned_grid(&large_built, target)?,
    );
    if small_grid >= large_grid {
        return Err(format!(
            "the two capacities must plan different grids, the larger the wider: {small_grid} splits at \
             {SMALL_CAPACITY} keys, {large_grid} at {LARGE_CAPACITY}"
        ));
    }
    for len in CAPACITY_LIVE_LENGTHS {
        let small_inputs = small.inputs(&small_built, &[len], true);
        let mut large_inputs = large.inputs(&large_built, &[len], true);
        // The small cache's first `len` key and value rows, copied into the large cache's layout.
        for (small_id, large_id, width) in [
            (small_built.k, large_built.k, small.dk),
            (small_built.v, large_built.v, small.dv),
        ] {
            let prefix = small_inputs.0[&small_id].as_f32().expect("f32 operand");
            let mut wide = large_inputs.0[&large_id]
                .as_f32()
                .expect("f32 operand")
                .to_vec();
            for head in 0..large.hkv {
                let (from, to) = (head * SMALL_CAPACITY * width, head * LARGE_CAPACITY * width);
                let live = len as usize * width;
                wide[to..to + live].copy_from_slice(&prefix[from..from + live]);
            }
            large_inputs.0.insert(
                large_id,
                HostTensor::f32(vec![large.batch, large.hkv, LARGE_CAPACITY, width], wide),
            );
        }
        let positions = [vec![len - 1]];
        let from_small = crate::live_attention::live_steps(
            exec,
            target,
            &small_built,
            &small_inputs,
            &positions,
        )
        .map_err(|e| format!("capacity {SMALL_CAPACITY} live length {len}: {e}"))?;
        let from_large = crate::live_attention::live_steps(
            exec,
            target,
            &large_built,
            &large_inputs,
            &positions,
        )
        .map_err(|e| format!("capacity {LARGE_CAPACITY} live length {len}: {e}"))?;
        let bits = |steps: &[Vec<f32>]| -> Vec<u32> {
            steps.iter().flatten().map(|x| x.to_bits()).collect()
        };
        if from_small[0].iter().any(|x| !x.is_finite()) {
            return Err(format!(
                "live length {len}: a non-finite output at capacity {SMALL_CAPACITY}: {:?}",
                from_small[0]
            ));
        }
        if bits(&from_small) != bits(&from_large) {
            return Err(format!(
                "live length {len}: the output at capacity {LARGE_CAPACITY} ({large_grid} splits) is not \
                 bit-identical to capacity {SMALL_CAPACITY} ({small_grid} splits)"
            ));
        }
    }
    Ok(())
}

// ---- Grouped-query decode (POOT-1549 row 14) ----

/// The live length of each step of the grouped-query rows: the full cache, a mid-length that ends inside a
/// key's cluster walk, and a one-key length.
const GQA_LIVE_LENGTHS: [i32; 3] = [512, 77, 1];

/// The grouped-query shapes of [`assert_gqa_decode_matches_the_oracle`], each at reduced length: the
/// Qwen3.8-27B attention shape (`n_rep = 6`, `dk = dv = 256`) and the qwen2.5 shape (`n_rep = 7`, `dk = 64`).
/// Two key/value heads each, so a query head that read another group's key head is wrong.
///
/// The operands are `f32`: the live graph builder fuses only an `f32` composite into the generated attention
/// (a two-byte one stays primitives, `attention_live.rs`), so an `f16` cache reaches this kernel only through a
/// model's typed KV state (the coherence rows) and the interpreter rows of `poot-kernelgen`.
fn gqa_fixtures() -> [poot_eval::test_support::live_attention::LiveAttentionFixture; 2] {
    let at = |n_rep: usize, width: usize| {
        poot_eval::test_support::live_attention::LiveAttentionFixture {
            batch: 1,
            hq: 2 * n_rep,
            hkv: 2,
            cap: 512,
            dk: width,
            dv: width,
            shared_dim: None,
            softcap: None,
            dtype: poot_tensor::DType::F32,
        }
    };
    [at(6, 256), at(7, 64)]
}

/// The grouped-query decode of [`gqa_fixtures`] stepped through [`GQA_LIVE_LENGTHS`] over one executable
/// on `exec` at `target`: the plan holds one generated attention, and every step's output, for every query
/// head of every key/value head's group, matches the CPU oracle's evaluation of that live length within
/// [`SPLIT_KV_DEVICE_TOLERANCE`]. The cache is full and one set of consts serves every step, so a step that
/// read past its live length changes the output.
pub fn assert_gqa_decode_matches_the_oracle(
    exec: &mut dyn Executor,
    target: poot_graph_plan::Target,
) -> Result<(), String> {
    for fixture in gqa_fixtures() {
        let label = format!(
            "n_rep {} dk {} dv {}",
            fixture.hq / fixture.hkv,
            fixture.dk,
            fixture.dv
        );
        let built = fixture.build(true);
        // The plan must hold the generated attention (at least one dispatch), not the primitives.
        let (dispatches, _) = attention_launches(&built, target);
        if dispatches == 0 {
            return Err(format!(
                "{label}: the live decode planned no attention dispatch"
            ));
        }
        let inputs = fixture.inputs(&built, &[GQA_LIVE_LENGTHS[0]], false);
        let positions: Vec<Vec<i32>> = GQA_LIVE_LENGTHS.iter().map(|&len| vec![len - 1]).collect();
        let got = crate::live_attention::live_steps(exec, target, &built, &inputs, &positions)
            .map_err(|e| format!("{label}: grouped-query run: {e}"))?;
        for (step, &len) in GQA_LIVE_LENGTHS.iter().enumerate() {
            let want = fixture.oracle(&inputs, &built, &[len]);
            if got[step].len() != want.len() {
                return Err(format!(
                    "{label}, live length {len}: {} elements against the oracle's {}",
                    got[step].len(),
                    want.len()
                ));
            }
            let worst = poot_test_util::max_abs_error(&got[step], &want);
            if worst > SPLIT_KV_DEVICE_TOLERANCE {
                return Err(format!(
                    "{label}, live length {len}: worst absolute error {worst} against the oracle exceeds {SPLIT_KV_DEVICE_TOLERANCE}"
                ));
            }
        }
    }
    Ok(())
}

// ---- Split-KV decode in each stage of a two-stage pipeline ----

/// Two split-KV decode layers of [`split_kv_fixture`]'s shape, one per pipeline stage: layer `l` attends
/// `q_l` over its own `k_l`/`v_l` cache with the live length `Slot::Pos` bounds, and layer 1 adds layer 0's
/// output to its own (the boundary value that crosses the stage cut). Adding rather than chaining keeps
/// each stage's result at full weight in the final output, so a stage that drops a split's partial moves
/// it by far more than the tolerance. Returns the graph and the named const tensors it binds.
fn split_kv_pipeline_graph() -> (poot_graph_ir::Graph, Vec<(&'static str, HostTensor)>) {
    use poot_graph_ir::ops::{AttentionKeys, attention_live, causal_mask_from_pos};
    use poot_graph_ir::{BinOp, IndexRange, Outputs, Scalar, Slot, ValidationId};
    use poot_test_util::fill;

    let f = split_kv_fixture();
    let (hq, hkv, cap, dk, dv) = (f.hq, f.hkv, f.cap, f.dk, f.dv);
    let b = Builder::for_tests();
    let pos = b
        .slot_ranged(
            Slot::Pos,
            TensorType::new(vec![1, 1], poot_tensor::DType::I32),
            IndexRange::F32_EXACT,
        )
        .unwrap();
    let mut tensors = Vec::new();
    let mut witnesses = Vec::new();
    let mut previous = None;
    for (layer, names) in [["q0", "k0", "v0"], ["q1", "k1", "v1"]]
        .into_iter()
        .enumerate()
    {
        let seed = 10 * layer as u64;
        let [qn, kn, vn] = names;
        let q = b.constant(qn, TensorType::f32(vec![1, hq, 1, dk])).unwrap();
        let k = b
            .constant(kn, TensorType::f32(vec![1, hkv, cap, dk]))
            .unwrap();
        let v = b
            .constant(vn, TensorType::f32(vec![1, hkv, cap, dv]))
            .unwrap();
        tensors.push((
            qn,
            HostTensor::f32(vec![1, hq, 1, dk], fill(hq * dk, 1 + seed)),
        ));
        tensors.push((
            kn,
            HostTensor::f32(vec![1, hkv, cap, dk], fill(hkv * cap * dk, 2 + seed)),
        ));
        tensors.push((
            vn,
            HostTensor::f32(vec![1, hkv, cap, dv], fill(hkv * cap * dv, 3 + seed)),
        ));
        let _layer = b.layer_scope(layer);
        let mask = causal_mask_from_pos(&b, pos, cap, None).unwrap();
        let len = b
            .binary_scalar(BinOp::Add, b.reshape(pos, vec![1]).unwrap(), Scalar::I32(1))
            .unwrap();
        let live = attention_live(
            &b,
            q,
            AttentionKeys::Single(k),
            v,
            hq / hkv,
            1.0 / (dk as f32).sqrt(),
            mask,
            None,
            len,
        )
        .expect("the live composite builds");
        witnesses.push((ValidationId(layer as u32 + 1), live.witness));
        previous = Some(match previous {
            Some(earlier) => b.binary(BinOp::Add, live.out, earlier).unwrap(),
            None => live.out,
        });
    }
    let result = previous.expect("two layers");
    let outputs = witnesses
        .into_iter()
        .fold(Outputs::new(result), |outputs, (id, witness)| {
            outputs.with_witness(id, format!("attention.len.{}", id.0), witness)
        });
    (
        b.finish(outputs).expect("the pipeline graph is valid"),
        tensors,
    )
}

/// The split-KV decode in both stages of a two-device layer pipeline: layer 0 on `devices[0]`, layer 1 on
/// `devices[1]`, each stage's attention must plan the split schedule on its own device's caps, and every
/// step of [`SPLIT_KV_LIVE_LENGTHS`] must equal the oracle's `attend(q0,k0,v0) + attend(q1,k1,v1)` within
/// [`SPLIT_KV_DEVICE_TOLERANCE`]. The executor is the one under test (`exec` loads the weights and takes
/// the staged program), `targets` names every device it owns.
pub fn assert_split_kv_pipeline_matches_the_oracle(
    exec: &mut dyn Executor,
    targets: &poot_graph_plan::staged::TargetSet,
    devices: [poot_graph_plan::multi_device::topology::DeviceId; 2],
) -> Result<(), String> {
    use poot_eval::test_support::live_attention::Inputs;
    use poot_executor::{NoSync, StepInputs};
    use poot_graph_ir::test_support::const_weights::ConstWeights;
    use poot_graph_ir::{Slot, Storage};
    use poot_graph_plan::{
        CompileLimits, CompileOptions, staged::DevicePlacement, staged::ExpertPlacement,
        staged::Partition, staged::compile_staged,
    };
    use poot_quant::StoredBytes;
    use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
    use poot_tensor::HostView;

    let (graph, tensors) = split_kv_pipeline_graph();
    let mut store = WeightStore::builder();
    let mut pos_key = None;
    for &id in graph.inputs() {
        let meta = graph.meta(id);
        match meta.storage {
            Storage::Const => {
                let name = meta.name().expect("a named const").into_owned();
                let tensor = &tensors
                    .iter()
                    .find(|(n, _)| *n == name)
                    .ok_or_else(|| format!("no tensor for const {name}"))?
                    .1;
                let dense = DenseWeight::try_new(
                    poot_tensor::DType::F32,
                    tensor.shape().to_vec(),
                    StoredBytes::from(tensor.view().bytes().to_vec()),
                )
                .map_err(|e| format!("{name}: {e}"))?;
                store
                    .insert(name, WeightEntry::Dense(dense))
                    .map_err(|e| format!("weights: {e}"))?;
            }
            Storage::Slot(Slot::Pos) => {
                pos_key = Some((meta.slot_key().unwrap().clone(), meta.aval.shape.clone()));
            }
            other => return Err(format!("unexpected pipeline input storage {other:?}")),
        }
    }
    let store = std::sync::Arc::new(store.build());
    let mut weights = ConstWeights::new(&store);
    let named = weights.graph(&graph);
    let program = compile_staged(
        &named,
        targets,
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Pipeline {
                after_layers: vec![0],
                devices: devices.to_vec(),
            },
        },
        &CompileOptions::new(CompileLimits::STANDARD)
            .with_matmul_precision(poot_graph_ir::MatmulPrecision::Exact),
    )
    .map_err(|e| format!("compile_staged: {e}"))?;
    let mut planned = Vec::new();
    for (_, device, stage) in program.stages() {
        planned.push((device, stage_attention_launches(stage)));
    }
    let want_plan: Vec<_> = devices.iter().map(|&d| (d, Some((2, 1)))).collect();
    if planned != want_plan {
        return Err(format!(
            "each stage must plan its one attention as the split plan (2, 1) on its device: {planned:?}"
        ));
    }
    let (key, shape) = pos_key.ok_or("the pipeline reads Slot::Pos")?;
    let exe = exec
        .load_weights(
            std::sync::Arc::clone(&store),
            std::sync::Arc::new(weights.map()),
        )
        .map_err(|e| format!("load_weights: {e}"))?;
    let entry = exec
        .add_entry(exe, &program)
        .map_err(|e| format!("add_entry: {e}"))?;
    let mut outcome = Ok(());
    let fixture = split_kv_fixture();
    let single = fixture.build(true);
    let tensor = |name: &str| {
        tensors
            .iter()
            .find(|(n, _)| *n == name)
            .expect("a pipeline tensor")
            .1
            .clone()
    };
    for &len in &SPLIT_KV_LIVE_LENGTHS {
        let pos_bytes = (len - 1).to_le_bytes();
        let mut step_in = StepInputs::new();
        step_in.push(
            key.clone(),
            &shape,
            HostView::new(poot_tensor::DType::I32, 1, &pos_bytes).unwrap(),
        );
        let got = exec
            .step(exe, entry, &step_in, &mut NoSync)
            .map_err(|e| format!("pipeline step at length {len}: {e}"))
            .and_then(|mut output| {
                output
                    .read()
                    .map(|bytes| {
                        bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|chunk| f32::from_le_bytes(*chunk))
                            .collect::<Vec<f32>>()
                    })
                    .map_err(|e| format!("read: {e}"))
            });
        let got = match got {
            Ok(got) => got,
            Err(e) => {
                outcome = Err(e);
                break;
            }
        };
        let mut want = vec![0.0f32; got.len()];
        for [qn, kn, vn] in [["q0", "k0", "v0"], ["q1", "k1", "v1"]] {
            let inputs = Inputs(std::collections::HashMap::from([
                (single.q, tensor(qn)),
                (single.k, tensor(kn)),
                (single.v, tensor(vn)),
            ]));
            let layer = fixture.oracle(&inputs, &single, &[len]);
            if layer.len() != want.len() {
                outcome = Err(format!(
                    "split-KV pipeline length {len}: {} elements against the oracle's {}",
                    got.len(),
                    layer.len()
                ));
                break;
            }
            for (w, l) in want.iter_mut().zip(&layer) {
                *w += l;
            }
        }
        if outcome.is_err() {
            break;
        }
        let worst = poot_test_util::max_abs_error(&got, &want);
        if worst > SPLIT_KV_DEVICE_TOLERANCE {
            outcome = Err(format!(
                "split-KV pipeline live length {len}: worst absolute error {worst} against the oracle exceeds {SPLIT_KV_DEVICE_TOLERANCE}"
            ));
            break;
        }
    }
    exec.remove_entry(exe, entry)
        .map_err(|e| format!("remove_entry: {e}"))?;
    exec.unload(exe).map_err(|e| format!("unload: {e}"))?;
    outcome
}

#[cfg(test)]
mod tests {
    use poot_graph_plan::Target;
    use poot_target::{AmdArch, Backend};
    use poot_test_util::device_caps::default_caps_for;

    use super::*;

    /// The device rows are only as good as the plan they run: on every backend's default caps the fixture's
    /// attention is the split plan (two launches, one scratch). A fixture the planner quietly stopped
    /// splitting would otherwise compare the streaming schedule with the oracle on a device.
    #[test]
    fn the_split_kv_fixture_plans_the_split_schedule() {
        let built = split_kv_fixture().build(true);
        for backend in [
            Backend::SpirvVulkan,
            Backend::AmdGcn(AmdArch::gfx1151()),
            Backend::Nvptx,
        ] {
            let target = Target::Gpu {
                backend,
                caps: default_caps_for(backend),
            };
            assert_eq!(attention_launches(&built, target), (2, 1), "{backend:?}");
        }
    }
}
