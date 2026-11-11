//! The stable rank kernel (POOT-1543) on the kernel-IR reference interpreter, no GPU: every row's rank against an
//! independent plain-Rust sort of `(class, canonical score, index)` and against the oracle's evaluation of the
//! op's primitive definition, compared exactly. Every row builds its body through `generate`, the one seam the
//! planner uses.
//!
//! Mutations (applied once to `stable_rank.rs`, observed red, restored): a tie broken towards the higher index;
//! no `-0` normalization (`order_key`); NaN not canonicalized (`order_key`); padding that sorts before the real
//! entries (pad key `u32::MAX`); the written position `p + 1`.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::test_support::stable_rank::{eligibility_rows, reference_rank, score_rows};
use poot_graph_ir::{Builder, Outputs, TensorType, ValueId, ops};
use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen::{
    BodyLimits, KernelGenError, KernelRequest, STABLE_RANK_MAX_LANES, StableRankSpec, generate,
};
use poot_tensor::HostTensor;

fn limits() -> BodyLimits {
    BodyLimits {
        max_instructions: NonZeroUsize::new(1 << 16).unwrap(),
        max_locals: NonZeroUsize::new(1 << 18).unwrap(),
    }
}

/// Run the kernel over `scores` (one row each) and, when masked, the eligibility rows.
fn interpret(scores: &[Vec<f32>], eligible: Option<&[Vec<f32>]>) -> Vec<Vec<f32>> {
    let experts = scores[0].len();
    let spec = StableRankSpec {
        rows: scores.len(),
        experts,
        masked: eligible.is_some(),
    };
    let generated = generate(
        &KernelRequest::StableRank(spec.clone()),
        &poot_test_util::device_caps::wgpu_rdna3_igpu(),
        &limits(),
    )
    .expect("the stable rank request generates");
    assert_eq!(generated.body.workgroup_size, [spec.lanes() as u32, 1, 1]);
    let mut buffers = vec![Buffer::from_f32s(&scores.concat())];
    if let Some(eligible) = eligible {
        buffers.push(Buffer::from_f32s(&eligible.concat()));
    }
    buffers.push(Buffer::from_f32s(&vec![f32::NAN; scores.len() * experts]));
    let grid = generated.launch.grid;
    run(
        &generated.body,
        [grid[0] / spec.lanes() as u32, grid[1], grid[2]],
        &mut buffers,
    )
    .unwrap();
    buffers
        .last()
        .unwrap()
        .to_f32s()
        .unwrap()
        .chunks(experts)
        .map(<[f32]>::to_vec)
        .collect()
}

/// The oracle's evaluation of the primitive definition the op stands for.
fn definition(scores: &[Vec<f32>], eligible: Option<&[Vec<f32>]>) -> Vec<Vec<f32>> {
    let (rows, e) = (scores.len(), scores[0].len());
    let b = Builder::for_tests();
    let ty = TensorType::f32(vec![rows, e]);
    let s = b.constant("scores", ty.clone()).unwrap();
    let m = eligible.map(|_| b.constant("eligible", ty.clone()).unwrap());
    let rank = match m {
        Some(m) => ops::stable_descending_rank_masked(&b, s, m).unwrap(),
        None => ops::stable_descending_rank(&b, s).unwrap(),
    };
    let graph = b.finish(Outputs::new(rank)).unwrap();
    let mut values: HashMap<ValueId, Value> = HashMap::from([(
        s.id(),
        HostTensor::f32(vec![rows, e], scores.concat()).into(),
    )]);
    if let (Some(m), Some(eligible)) = (m, eligible) {
        values.insert(
            m.id(),
            HostTensor::f32(vec![rows, e], eligible.concat()).into(),
        );
    }
    let out = eval(&graph, &values, EvalOptions::default()).unwrap();
    out.output
        .into_host()
        .unwrap()
        .as_f32()
        .unwrap()
        .chunks(e)
        .map(<[f32]>::to_vec)
        .collect()
}

/// `got` equals `want` row by row, bit for bit; the failure names the row and the first element that differs.
fn assert_ranks(label: &str, got: &[Vec<f32>], want: &[Vec<f32>], scores: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len(), "{label}: row count");
    for (r, (got, want)) in got.iter().zip(want).enumerate() {
        if let Some(i) = got
            .iter()
            .zip(want)
            .position(|(g, w)| g.to_bits() != w.to_bits())
        {
            panic!(
                "{label}: row {r} element {i} (score {}): got {}, want {}",
                scores[r][i], got[i], want[i]
            );
        }
    }
}

fn assert_ranks_like_the_definition(
    label: &str,
    scores: &[Vec<f32>],
    eligible: Option<&[Vec<f32>]>,
) {
    let got = interpret(scores, eligible);
    let plain: Vec<Vec<f32>> = scores
        .iter()
        .enumerate()
        .map(|(r, row)| reference_rank(row, eligible.map(|m| m[r].as_slice())))
        .collect();
    assert_ranks(&format!("{label} (plain sort)"), &got, &plain, scores);
    assert_ranks(
        &format!("{label} (the definition)"),
        &got,
        &definition(scores, eligible),
        scores,
    );
}

/// Extents around every power of two the network pads to (1 and 2 need one stage or none, 5 and 31 pad to 8 and
/// 32, 512 is exactly full, 513 and 1000 pad to 1024 so a lane owns several pairs), rows around the fold, all
/// compared with the plain sort and the definition: descending and ascending runs, an all-equal row, ties across
/// the padding boundary, signed zeros, NaN, infinities and the finite extremes.
#[test]
fn the_kernel_ranks_what_the_definition_ranks() {
    for e in [1, 2, 3, 5, 31, 32, 33, 512, 513, 1000] {
        let rows = score_rows(e);
        assert_ranks_like_the_definition(&format!("E={e}"), &rows, None);
    }
}

/// Rows past one workgroup of a fold, so every row index reaches its own row.
#[test]
fn rows_are_independent() {
    for rows in [3usize, 129] {
        let e = 40;
        let scores: Vec<Vec<f32>> = (0..rows)
            .map(|r| {
                score_rows(e)[r % 6]
                    .iter()
                    .map(|x| x + (r / 6) as f32)
                    .collect()
            })
            .collect();
        assert_ranks_like_the_definition(&format!("{rows} rows"), &scores, None);
    }
}

/// The masked rank puts the eligible class (`>= 1`) first, then the same order inside each class; `0.5` and NaN
/// are not eligible.
#[test]
fn the_masked_kernel_ranks_what_the_definition_ranks() {
    for e in [1, 2, 5, 31, 512, 513] {
        let scores = score_rows(e);
        let masks = eligibility_rows(scores.len(), e);
        assert_ranks_like_the_definition(&format!("masked E={e}"), &scores, Some(&masks));
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
    let dir = std::env::temp_dir().join("poot-stable-rank-lowering");
    std::fs::create_dir_all(&dir).unwrap();
    for masked in [false, true] {
        let generated = generate(
            &KernelRequest::StableRank(StableRankSpec {
                rows: 3,
                experts: 300,
                masked,
            }),
            &poot_test_util::device_caps::wgpu_rdna3_igpu(),
            &limits(),
        )
        .unwrap();
        for target in [Target::SpirvVulkan, Target::Nvptx] {
            let out = poot_codegen::artifact_path(&dir, &format!("stable_rank_{masked}"), target);
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

/// The geometry the planner reads: half the padded entries as lanes up to the cap, two words per entry.
#[test]
fn the_spec_names_its_lanes_and_workgroup_memory() {
    let spec = |experts| StableRankSpec {
        rows: 1,
        experts,
        masked: false,
    };
    assert_eq!((spec(1).lanes(), spec(1).lds_bytes()), (1, 16));
    assert_eq!((spec(5).lanes(), spec(5).lds_bytes()), (4, 64));
    assert_eq!((spec(512).lanes(), spec(512).lds_bytes()), (256, 4096));
    assert_eq!(
        (spec(2048).lanes(), spec(2048).lds_bytes()),
        (STABLE_RANK_MAX_LANES, 16384)
    );
}

/// An empty extent and an extent past the exactly representable ranks are refused by the generator, not run.
#[test]
fn an_extent_no_rank_can_name_is_refused() {
    let refused = |rows, experts| {
        generate(
            &KernelRequest::StableRank(StableRankSpec {
                rows,
                experts,
                masked: false,
            }),
            &poot_test_util::device_caps::wgpu_rdna3_igpu(),
            &limits(),
        )
    };
    assert!(matches!(
        refused(1, 0),
        Err(KernelGenError::BelowMinimum { ref what, .. }) if what == "experts"
    ));
    assert!(matches!(
        refused(0, 8),
        Err(KernelGenError::BelowMinimum { ref what, .. }) if what == "rows"
    ));
    assert!(matches!(
        refused(1, (1 << 24) + 1),
        Err(KernelGenError::ExceedsBound { ref what, .. }) if what == "experts"
    ));
}
