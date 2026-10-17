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
    splits: Option<usize>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Vec<f32> {
    let d = case.d;
    let spec = kg::AttentionSpec {
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
        },
        bound: kg::KeyBound::Static,
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

fn assert_matches_reference(label: &str, case: &Case, lanes: usize, splits: Option<usize>) {
    let [q, k, v] = case.operands(1);
    let got = interpret(case, lanes, splits, &q, &k, &v);
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
/// selection of every complete block (the dense causal case), at the lane counts of the planner.
#[test]
fn the_kernel_attends_over_exactly_the_selected_blocks() {
    for case in cases() {
        for lanes in [8, 16, 64] {
            assert_matches_reference(
                &format!("{} on {lanes} lanes", case.label),
                &case,
                lanes,
                None,
            );
        }
        // The candidates divided across workgroups, the last chunk ragged and (with few candidates) empty.
        for splits in [2, 3, 5] {
            assert_matches_reference(
                &format!("{} on {splits} splits", case.label),
                &case,
                8,
                Some(splits),
            );
        }
    }
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
    assert_matches_reference("stray id", &case, 8, None);
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
    assert_matches_reference("future block", &case, 8, None);
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
        eprintln!("llc not on PATH; skipping");
        return;
    }
    let dir = std::env::temp_dir().join("poot-selected-attention-lowering");
    std::fs::create_dir_all(&dir).unwrap();
    let spec = kg::AttentionSpec {
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
        mask: kg::MaskKind::Blocks { ratio: 4, k: 8 },
        bound: kg::KeyBound::Static,
        schedule: kg::AttentionSchedule::Streaming { lanes: 64 },
    };
    let generated = generate_request(&spec);
    for target in [Target::SpirvVulkan, Target::Nvptx] {
        let out = poot_codegen::artifact_path(&dir, "selected_attention", target);
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
