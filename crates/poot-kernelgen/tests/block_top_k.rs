//! The block top-k kernel (POOT-1143 stage 2) on the kernel-IR reference interpreter, no GPU: the selected
//! block set of every query against an independent plain-Rust selection (stable sort of the eligible
//! blocks by canonical score, lower block on ties). The ids, the `-1` padding and the keep indicator are
//! compared exactly. Every row builds its body through `generate`, the one seam the planner uses.
//!
//! Mutations (applied once to `block_top_k.rs`, observed red, restored): tie placement taking the last
//! equal blocks instead of the first; the radix bin off by one (`need` not reduced by the bins above);
//! eligibility `(pos + 1) / ratio` read as `pos / ratio`; no NaN canonicalization.

use std::num::NonZeroUsize;

use poot_graph_ir::test_support::block_top_k::{grid_scores, reference_row, selection_cases};
use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen::{
    BLOCK_TOP_K_LANES, BlockTopKSpec, BodyLimits, KernelGenError, KernelRequest, generate,
};
use poot_target::DeviceCaps;

fn limits() -> BodyLimits {
    BodyLimits {
        max_instructions: NonZeroUsize::new(1 << 16).unwrap(),
        max_locals: NonZeroUsize::new(1 << 18).unwrap(),
    }
}

/// Run the kernel over `queries` and return the packed rows, `[k + blocks]` per query.
fn interpret(scores: &[Vec<f32>], pos: &[i32], ratio: usize, k: usize) -> Vec<Vec<i32>> {
    let blocks = scores[0].len();
    let spec = BlockTopKSpec {
        queries: scores.len(),
        blocks,
        ratio,
        k,
    };
    let generated = generate(
        &KernelRequest::BlockTopK(spec),
        &DeviceCaps::wgpu_rdna3_igpu(),
        &limits(),
    )
    .expect("the block top-k request generates");
    assert_eq!(
        generated.body.workgroup_size,
        [BLOCK_TOP_K_LANES as u32, 1, 1]
    );
    let mut buffers = vec![
        Buffer::from_f32s(&scores.concat()),
        Buffer::from_i32s(pos),
        Buffer::from_i32s(&vec![i32::MIN; scores.len() * (k + blocks)]),
    ];
    let grid = generated.launch.grid;
    run(
        &generated.body,
        [grid[0] / BLOCK_TOP_K_LANES as u32, grid[1], grid[2]],
        &mut buffers,
    )
    .unwrap();
    buffers[2]
        .to_i32s()
        .unwrap()
        .chunks(k + blocks)
        .map(<[i32]>::to_vec)
        .collect()
}

fn assert_selects(label: &str, scores: &[Vec<f32>], pos: &[i32], ratio: usize, k: usize) {
    let got = interpret(scores, pos, ratio, k);
    for (q, row) in got.iter().enumerate() {
        assert_row(
            &format!("{label}: query {q} at position {}", pos[q]),
            row,
            &reference_row(&scores[q], pos[q], ratio, k),
            k,
        );
    }
}

/// `got` equals `want` word for word; the failure names the first word that differs: an id slot or a block's
/// keep indicator.
fn assert_row(label: &str, got: &[i32], want: &[i32], k: usize) {
    assert_eq!(got.len(), want.len(), "{label}: row length");
    if let Some(at) = got.iter().zip(want).position(|(g, w)| g != w) {
        let what = if at < k {
            format!("id slot {at}")
        } else {
            format!("keep indicator of block {}", at - k)
        };
        panic!("{label}: {what}: got {}, want {}", got[at], want[at]);
    }
}

/// SC-1143-8, the selection stage on identical input (design section 5, tier 1): ties (every score equal, so
/// the lower blocks win), ties straddling the threshold, fewer eligible blocks than `k`, `E == k`,
/// `E == k + 1`, NaN, infinities and `+/-0.0`, a block count that is not a power of two. Each row is
/// compared with the plain definition for ids, padding and indicator.
#[test]
fn the_kernel_selects_what_the_definition_selects() {
    for (label, scores, positions, ratio, k) in selection_cases() {
        assert_selects(&label, &scores, &positions, ratio, k);
    }
}

/// A block count that is neither a power of two nor a multiple of the lanes, wide enough that every lane
/// owns several blocks, with and without ties.
#[test]
fn an_awkward_block_count_selects_what_the_definition_selects() {
    let blocks = 1237;
    let last = 4 * blocks as i32 - 1;
    for levels in [1 << 20, 16, 1] {
        let scores = vec![grid_scores(blocks, levels, 3); 3];
        assert_selects("awkward", &scores, &[last, last / 2, 4 * 700 + 1], 4, 100);
    }
}

/// The block counts at which a register-resident row would change its registers per lane (16384, 32768
/// and 65536 blocks), and a tied row at each: the kernel re-reads global memory, so none of them is a
/// special case, and this keeps it so.
#[test]
fn the_kernel_selects_at_16384_32768_and_65536_blocks() {
    for blocks in [16384usize, 32768, 65536] {
        let scores = vec![grid_scores(blocks, 64, 1), vec![0.5; blocks]];
        let last = 4 * blocks as i32 - 1;
        assert_selects("wide", &scores, &[last, last], 4, 512);
    }
}

/// The generated body lowers to a real SPIR-V module and PTX through `llc` (skipped when `llc` is absent, as
/// `llc.rs` does): the device rows run these artifacts, and a construct a backend cannot lower fails here
/// with the compiler's own message before any device is opened.
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
    let dir = std::env::temp_dir().join("poot-block-top-k-lowering");
    std::fs::create_dir_all(&dir).unwrap();
    let generated = generate(
        &KernelRequest::BlockTopK(BlockTopKSpec {
            queries: 3,
            blocks: 300,
            ratio: 4,
            k: 50,
        }),
        &DeviceCaps::wgpu_rdna3_igpu(),
        &limits(),
    )
    .unwrap();
    for target in [Target::SpirvVulkan, Target::Nvptx] {
        let out = poot_codegen::artifact_path(&dir, "block_top_k", target);
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

/// The counts are packed eight bits to a count while no lane can own more than 255 blocks, sixteen after:
/// 65280 blocks is 255 per lane, and an all-equal row puts every one of them in one digit, the largest
/// count the narrow width must hold; one block more switches width. A count that carried into its
/// neighbour would corrupt the threshold, so both rows must still select the first `k` blocks.
///
/// Mutation (observed red, restored): the narrow width admitted up to 256 owned blocks, so the all-equal
/// row's count wraps into the next digit's count.
#[test]
fn a_lane_at_the_widest_count_of_its_width_does_not_carry() {
    for blocks in [65280usize, 65281] {
        let last = 4 * blocks as i32 - 1;
        assert_selects("widest count", &[vec![0.5; blocks]], &[last], 4, 512);
    }
}

/// A block count no packed width holds is refused by the generator, with the bound, not run.
#[test]
fn a_block_count_no_count_width_holds_is_refused() {
    let blocks = BLOCK_TOP_K_LANES * u16::MAX as usize + 1;
    let refused = generate(
        &KernelRequest::BlockTopK(BlockTopKSpec {
            queries: 1,
            blocks,
            ratio: 4,
            k: 512,
        }),
        &DeviceCaps::wgpu_rdna3_igpu(),
        &limits(),
    );
    match refused {
        Err(KernelGenError::ExceedsBound { what, value, .. }) => {
            assert_eq!((what.as_str(), value), ("blocks", blocks));
        }
        other => panic!("expected the block count refused, got {other:?}"),
    }
}
