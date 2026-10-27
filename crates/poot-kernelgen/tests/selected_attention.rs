//! The block-selected attention kernel (POOT-1143 stage 3) on the kernel-IR reference interpreter, no GPU:
//! against an independent f64 attention over exactly the keys the selection names (its selected blocks and the
//! query's own unfinished block), and against the dense attention under the mask those keys make. Every row
//! builds its body through `generate`, the one seam the planner uses.
//!
//! Mutations (applied once to `attention.rs`, observed red, restored): the unfinished block's start read as
//! `pos / ratio * ratio` (one block early); the `-1` id padding read as block 0; the causal bound dropped;
//! the selected block's offset dropped (every position of a block read as its first).

use std::num::NonZeroUsize;

use poot_graph_ir::test_support::selected_attention::{Case, cases};
use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen as kg;
use poot_target::DeviceCaps;

fn limits() -> kg::BodyLimits {
    kg::BodyLimits {
        max_instructions: NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: NonZeroUsize::new(1 << 20).unwrap(),
    }
}

/// Run the generated kernel on the interpreter and return its output: the streaming schedule, or with `splits`
/// the split-KV stage one over the candidates and the combine after it (the scratch arrives poisoned with NaN,
/// as an arena slot's stale contents).
fn interpret(
    case: &Case,
    lanes: usize,
    grouped: bool,
    splits: Option<usize>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Vec<f32> {
    let d = case.d;
    let spec = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        batch: case.batch,
        hq: case.hq,
        n_rep: case.n_rep,
        lq: case.lq,
        lk: case.lk,
        dk: d,
        dv: d,
        key: kg::KeyLayout::Single,
        score: kg::ScoreTransform::Scale {
            scale_bits: (1.0 / (d as f32).sqrt()).to_bits(),
        },
        mask: kg::MaskKind::Blocks {
            ratio: case.ratio,
            k: case.k,
            grouped,
        },
        bound: kg::KeyBound::Static,
        sink: kg::AttentionSink::Absent,
        schedule: match splits {
            None => kg::AttentionSchedule::Streaming { lanes },
            Some(splits) => kg::AttentionSchedule::SplitKv { lanes, splits },
        },
    };
    let generated = generate_request(&spec);
    let out_len = case.batch * case.hq * case.lq * d;
    let groups = |launch: &kg::Launch| {
        [
            launch.grid[0] / lanes as u32,
            launch.grid[1],
            launch.grid[2],
        ]
    };
    let mut buffers = vec![
        Buffer::from_f32s(q),
        Buffer::from_f32s(k),
        Buffer::from_f32s(v),
        Buffer::from_i32s(&case.ids),
        Buffer::from_i32s(&case.pos),
    ];
    match splits {
        None => {
            buffers.push(Buffer::from_f32s(&vec![f32::NAN; out_len]));
            run(&generated.body, groups(&generated.launch), &mut buffers).unwrap();
            buffers[5].to_f32s().unwrap()
        }
        Some(_) => {
            let combine_spec =
                kg::AttentionCombineSpec::of(&spec).expect("a split schedule combines");
            let combine = kg::generate(
                &kg::KernelRequest::AttentionCombine(combine_spec),
                &DeviceCaps::wgpu_rdna3_igpu(),
                &limits(),
            )
            .expect("the combine generates");
            buffers.push(Buffer::from_f32s(&vec![
                f32::NAN;
                spec.partial_elems()
                    .expect("partials")
            ]));
            run(&generated.body, groups(&generated.launch), &mut buffers).unwrap();
            let mut combine_buffers = vec![
                std::mem::replace(&mut buffers[5], Buffer::from_f32s(&[])),
                Buffer::from_f32s(&vec![f32::NAN; out_len]),
            ];
            run(&combine.body, groups(&combine.launch), &mut combine_buffers).unwrap();
            combine_buffers[1].to_f32s().unwrap()
        }
    }
}

fn generate_request(spec: &kg::AttentionSpec) -> kg::Generated {
    kg::generate(
        &kg::KernelRequest::Attention(spec.clone()),
        &DeviceCaps::wgpu_rdna3_igpu(),
        &limits(),
    )
    .expect("the selected attention request generates")
}

fn assert_matches_reference(
    label: &str,
    case: &Case,
    (lanes, grouped): (usize, bool),
    splits: Option<usize>,
) {
    let [q, k, v] = case.operands(1);
    let got = interpret(case, lanes, grouped, splits, &q, &k, &v);
    let want = case.reference(&q, &k, &v);
    assert_eq!(got.len(), want.len(), "{label}: output length");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (f64::from(*g) - w).abs() <= 1e-5 * (1.0 + w.abs()),
            "{label}: output element {i} (row {}, channel {}): got {g}, reference {w}",
            i / case.d,
            i % case.d
        );
    }
}

/// SC-1143-10, the kernel against the independent selection-restricted softmax (design section 5, tier 2):
/// decode and prefill queries, GQA, positions that put the query on a block boundary, inside a block and before
/// the first block completes (an empty selection), `-1` padding after the last selected block, and a
/// selection of every complete block (the dense causal case). A workgroup serves the `n_rep` heads of a
/// key/value head (POOT-1338), `slices` lanes each walking a share of the candidates: one slice (the
/// planner's width at a wide head), several, and more slices than a short row has candidates.
#[test]
fn the_kernel_attends_over_exactly_the_selected_blocks() {
    for case in cases() {
        // One head per workgroup (the planner's schedule while the rows alone fill the device).
        for lanes in [8, 16, 64] {
            assert_matches_reference(
                &format!("{} on {lanes} lanes", case.label),
                &case,
                (lanes, false),
                None,
            );
        }
        for slices in [1, 2, 5, 16] {
            assert_matches_reference(
                &format!("{} grouped on {slices} slices", case.label),
                &case,
                (case.n_rep * slices, true),
                None,
            );
        }
        // The candidates divided across workgroups, the last chunk ragged and (with few candidates) empty.
        for splits in [2, 3, 5] {
            for (lanes, grouped) in [(8, false), (case.n_rep * 2, true)] {
                assert_matches_reference(
                    &format!("{} on {splits} splits, grouped {grouped}", case.label),
                    &case,
                    (lanes, grouped),
                    Some(splits),
                );
            }
        }
    }
}

/// A grouped step reads the largest of 4, 2 and 1 keys of a block that divides it: blocks of 3 (one key), 6 (two)
/// and 8 (four) all equal the reference, the unfinished block ragged inside a step. Mutation: let a step read
/// four keys whatever the block size; the blocks of 3 and 6 read past their block and go red.
#[test]
fn a_grouped_step_never_crosses_a_block() {
    for ratio in [3, 6, 8] {
        let (blocks, k) = (6, 3);
        let lk = ratio * blocks;
        let pos: Vec<i32> = vec![0, 2, 7, 11, 13, lk as i32 - 1, 5, 9, 16, lk as i32 - 2];
        let case = Case {
            label: format!("ratio {ratio}"),
            batch: 2,
            hq: 6,
            n_rep: 3,
            lq: 5,
            lk,
            d: 8,
            ratio,
            k,
            ids: poot_graph_ir::test_support::selected_attention::selected_ids(
                &pos,
                blocks,
                ratio,
                k,
                1 << 20,
            ),
            pos,
        };
        for slices in [1, 3] {
            assert_matches_reference(
                &format!("ratio {ratio} on {slices} slices"),
                &case,
                (case.n_rep * slices, true),
                None,
            );
        }
    }
}

/// A head width that is a multiple of the 32 workgroup-memory banks swizzles each lane's partial `o` by the lane
/// (`channel ^ lane % 32`, see `OSlots`): the output equals the reference at 64 channels with more than 32 lanes
/// (the lane index wraps the 32 banks) and with a combine over slices that read other lanes' swizzled
/// partials. Mutation: swizzle by `lane` instead of `lane % 32`; the channels of lane 33 and up leave
/// their 32-aligned group and the rows go red.
#[test]
fn a_bank_swizzled_partial_equals_the_reference() {
    let (ratio, blocks, k) = (4, 6, 3);
    let lk = ratio * blocks;
    let pos: Vec<i32> = vec![3, 9, 14, 20, 23, 7, 17];
    let case = Case {
        label: "d 64".into(),
        batch: 1,
        hq: 4,
        n_rep: 2,
        lq: pos.len(),
        lk,
        d: 64,
        ratio,
        k,
        ids: poot_graph_ir::test_support::selected_attention::selected_ids(
            &pos,
            blocks,
            ratio,
            k,
            1 << 20,
        ),
        pos,
    };
    for (lanes, grouped) in [(40, false), (64, false), (2 * 20, true), (2 * 33, true)] {
        assert_matches_reference(
            &format!("d 64 on {lanes} lanes, grouped {grouped}"),
            &case,
            (lanes, grouped),
            None,
        );
    }
    assert_matches_reference("d 64 split", &case, (2 * 20, true), Some(3));
}

/// The stored key extent past the last block (a cache padded off the channel period stores one position
/// more) and an `ids` entry past the cache are never read: the output equals the one over the real keys.
#[test]
fn a_padded_cache_and_a_stray_id_read_no_key_past_the_stored_extent() {
    let (ratio, blocks) = (4, 4);
    let case = Case {
        label: "stray id".into(),
        batch: 1,
        hq: 2,
        n_rep: 1,
        lq: 1,
        lk: ratio * blocks + 1,
        d: 4,
        ratio,
        k: 3,
        // block 99 does not exist: its positions are past the stored keys
        ids: vec![0, 2, 99],
        pos: vec![ratio as i32 * blocks as i32 - 1],
    };
    assert_matches_reference("stray id", &case, (8, false), None);
}

/// An `ids` entry naming a block the query has not reached (its positions are after the query) reads no key
/// after the query: the kernel is causal on its own, whatever the selection holds. Mutation: drop the causal
/// bound in `block_candidate`.
#[test]
fn an_id_naming_a_future_block_reads_no_key_after_the_query() {
    let case = Case {
        label: "future block".into(),
        batch: 1,
        hq: 2,
        n_rep: 1,
        lq: 1,
        lk: 20,
        d: 4,
        ratio: 4,
        k: 3,
        // position 9: blocks 0 and 1 are complete, block 2 is the unfinished one; block 3 is in the future
        ids: vec![0, 3, 4],
        pos: vec![9],
    };
    assert_matches_reference("future block", &case, (8, false), None);
}

/// The generated body lowers to a real SPIR-V module that `spirv-val` accepts and to PTX through `llc`
/// (skipped when the tools are absent, as `llc.rs` does): a construct a backend cannot lower fails here with
/// the compiler's own message before any device is opened.
#[test]
fn the_kernel_lowers_to_spirv_and_ptx() {
    use poot_codegen::{Target, compile};
    if std::process::Command::new("llc")
        .arg("--version")
        .output()
        .is_err()
    {
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join("poot-selected-attention-lowering");
    std::fs::create_dir_all(&dir).unwrap();
    for grouped in [false, true] {
        let spec = kg::AttentionSpec {
            kv: poot_kernelgen::KvLane::F32,
            batch: 1,
            hq: 4,
            n_rep: 2,
            lq: 2,
            lk: 64,
            dk: 32,
            dv: 32,
            key: kg::KeyLayout::Single,
            score: kg::ScoreTransform::Scale {
                scale_bits: 0.25f32.to_bits(),
            },
            mask: kg::MaskKind::Blocks {
                ratio: 4,
                k: 8,
                grouped,
            },
            bound: kg::KeyBound::Static,
            sink: kg::AttentionSink::Absent,
            schedule: kg::AttentionSchedule::Streaming { lanes: 64 },
        };
        let generated = generate_request(&spec);
        for target in [Target::SpirvVulkan, Target::Nvptx] {
            let out = poot_codegen::artifact_path(
                &dir,
                &format!("selected_attention_grouped_{grouped}"),
                target,
            );
            compile(&generated.body, target, &out).unwrap_or_else(|e| panic!("{target:?}: {e}"));
            assert!(
                std::fs::metadata(&out).unwrap().len() > 0,
                "{target:?}: empty artifact"
            );
            if target == Target::SpirvVulkan
                && std::process::Command::new("spirv-val")
                    .arg("--version")
                    .output()
                    .is_ok()
            {
                let verdict = std::process::Command::new("spirv-val")
                    .args(["--relax-block-layout", "--target-env", "vulkan1.2"])
                    .arg(&out)
                    .output()
                    .unwrap();
                assert!(
                    verdict.status.success(),
                    "spirv-val rejects the module {}: {}",
                    out.display(),
                    String::from_utf8_lossy(&verdict.stderr)
                );
            }
        }
    }
}
