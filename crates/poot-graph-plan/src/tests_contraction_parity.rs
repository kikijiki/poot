//! One source per contraction (POOT-1237). The decode GEMV and the tiled GEMM each come from kernelgen's
//! `generate`; a launch-size trigger (the display-watchdog split, the RADV tile ceiling) changes how many
//! dispatches cover the output, never the body that computes it. These rows plan through the production
//! entry (`plan_eqn`), run every planned dispatch on the reference interpreter, and compare across the old
//! trigger boundary: the unsplit plan and the split plan of one shape must agree bit for bit, and with the
//! plain matmul oracle.

use poot_graph_ir::{Builder, Graph, TensorType};
use poot_kernel_ir::interp::{self, Buffer};
use poot_kernelgen::{ContractionSpec, DenseTiledContraction, DenseTiledWeight, KernelRequest};
use poot_target::{Backend, DeviceCaps, WatchdogBudget};

use crate::predicates::GEMV_WIDTH;
use crate::test_support::plan_eqn;
use crate::{Dispatch, KernelChoice, Plan};

fn fill(seed: u64, count: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 0.4 - 0.2
        })
        .collect()
}

/// A graph of one `MatMul` over f32 constants of the given shapes, and its only equation's plan on `caps`.
fn matmul_plan(a: &[usize], b: &[usize], backend: Backend, caps: &DeviceCaps) -> Plan {
    let bld = Builder::for_tests();
    let x = bld.constant("x", TensorType::f32(a.to_vec())).unwrap();
    let w = bld.constant("w", TensorType::f32(b.to_vec())).unwrap();
    let y = bld.matmul(x, w).unwrap();
    let g: Graph = bld.finish(y).unwrap();
    plan_eqn(&g, &g.eqns()[0], backend, caps).unwrap()
}

/// Run every dispatch of `plan` over `[x, w, meta] -> out` on the interpreter, in plan order, into one output
/// buffer; a dispatch whose body bakes its dims binds no metadata buffer.
fn run_plan(plan: &Plan, x: &[f32], w: &[f32], out_len: usize) -> Vec<f32> {
    let mut out = Buffer::from_f32s(&vec![0.0; out_len]);
    for dispatch in plan.dispatches() {
        let meta = &dispatch.launch().meta;
        let mut buffers = vec![Buffer::from_f32s(x), Buffer::from_f32s(w)];
        if !meta.is_empty() {
            buffers.push(Buffer::from_u32s(meta));
        }
        buffers.push(out);
        let body = dispatch.body();
        assert_eq!(body.params().count(), buffers.len(), "[x, w, meta?, out]");
        let grid = dispatch.launch().grid;
        let wg = body.workgroup_size;
        let groups = [
            grid[0].div_ceil(wg[0]),
            grid[1].div_ceil(wg[1]),
            grid[2].div_ceil(wg[2]),
        ];
        interp::run(body, groups, &mut buffers).expect("the planned body runs");
        out = buffers.pop().unwrap();
    }
    out.to_f32s().unwrap()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Row-major `[batch, m, k] @ [batch, k, n]` (the weight shared when `shared`), f32.
fn matmul_oracle(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    x: &[f32],
    w: &[f32],
    shared: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; batch * m * n];
    for b in 0..batch {
        let w_base = if shared { 0 } else { b * k * n };
        for r in 0..m {
            for c in 0..n {
                out[(b * m + r) * n + c] = (0..k)
                    .map(|i| x[(b * m + r) * k + i] * w[w_base + i * n + c])
                    .sum();
            }
        }
    }
    out
}

fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-4 + 1e-4 * w.abs();
        assert!((g - w).abs() <= tol, "{ctx}[{i}]: {g} vs {w}");
    }
}

/// A device whose X grid holds `max_x` workgroups and whose watchdog splits a decode GEMV above `trigger`
/// output elements: the production trigger, shrunk so the interpreter crosses it in a few workgroups.
fn watchdog_caps(max_x: u32, trigger: u64) -> DeviceCaps {
    let base = DeviceCaps::wgpu_rdna3_igpu();
    DeviceCaps {
        max_grid: [max_x, base.max_grid[1], base.max_grid[2]],
        watchdog_budget: Some(WatchdogBudget {
            decode_gemv_out_elems: trigger,
            ..base
                .watchdog_budget
                .expect("the wgpu iGPU fixture has a watchdog budget")
        }),
        ..base
    }
}

fn gemv_chunk(dispatch: &Dispatch) -> (usize, usize, usize, usize, usize) {
    match dispatch.choice() {
        KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::GemvChunk {
            k,
            n,
            width,
            elems,
            offset,
        })) => (*k, *n, *width, *elems, *offset),
        other => panic!("a decode GEMV is the generated gemv_lds body, got {other:?}"),
    }
}

/// SC-002: the decode GEMV across the old watchdog-trigger boundary. At `trigger` output elements the plan is
/// one dispatch; one element past it the plan is several, and every dispatch of both is the same generated
/// `GemvChunk` body over the same `[K, N]` weight. The split result equals the unsplit result of the same shape
/// bit for bit (same body, same summation order), single-sequence and batched.
///
/// Mutation: in `decode_gemv_plan`, plan the over-trigger branch as a different body (the serial matmul
/// kernel); the split plan's dispatches are no `GemvChunk`, the choice assertion names it, and the results
/// differ in the last bits.
#[test]
fn decode_gemv_is_one_body_on_both_sides_of_the_watchdog_boundary() {
    const TRIGGER: u64 = 100;
    let backend = Backend::SpirvVulkan;
    let split_caps = watchdog_caps(50, TRIGGER);
    let unsplit_caps = DeviceCaps {
        watchdog_budget: None,
        ..split_caps
    };
    let k = 37;
    for (batch, n) in [
        (1usize, 100usize),
        (1, 101),
        (1, 237),
        (3, 33),
        (3, 34),
        (4, 77),
    ] {
        let out_numel = batch * n;
        let x = fill(1, batch * k);
        let w = fill(2, k * n);
        let plan = matmul_plan(&[batch, 1, k], &[k, n], backend, &split_caps);
        let unsplit = matmul_plan(&[batch, 1, k], &[k, n], backend, &unsplit_caps);
        let crosses = out_numel as u64 > TRIGGER;
        assert_eq!(
            plan.dispatches().len() > 1,
            crosses,
            "B={batch} N={n}: {out_numel} outputs against a trigger of {TRIGGER}"
        );
        assert_eq!(
            unsplit.dispatches().len(),
            1,
            "no watchdog budget: one dispatch"
        );
        let mut covered = 0;
        for dispatch in plan.dispatches().iter().chain(unsplit.dispatches()) {
            let (gk, gn, width, _, _) = gemv_chunk(dispatch);
            assert_eq!((gk, gn, width), (k, n, GEMV_WIDTH), "one body per shape");
        }
        for dispatch in plan.dispatches() {
            let (_, _, _, elems, offset) = gemv_chunk(dispatch);
            assert_eq!(offset, covered, "ranges tile the output in order");
            covered += elems;
        }
        assert_eq!(covered, out_numel, "the ranges cover the whole output");

        let got = run_plan(&plan, &x, &w, out_numel);
        let whole = run_plan(&unsplit, &x, &w, out_numel);
        assert_eq!(
            bits(&got),
            bits(&whole),
            "B={batch} N={n}: split vs unsplit"
        );
        assert_close(
            &got,
            &matmul_oracle(batch, 1, k, n, &x, &w, true),
            &format!("B={batch} N={n}"),
        );
    }
}

/// The decode GEMV's planned grid is the one fold the body's `col = GroupY * x_groups + GroupX` reads: X fills
/// to the device's cap and the rest spills onto Y, one workgroup of `GEMV_WIDTH` lanes per output element.
///
/// Mutation: in `ContractionSpec::GemvChunk`'s `generate` arm, launch `[x * width, 1, 1]` (drop the Y spill);
/// N = 237 on a 50-wide grid plans a grid of one row and the assertion names the planned grid.
#[test]
fn decode_gemv_grid_spills_onto_y_past_the_x_cap() {
    let caps = DeviceCaps {
        watchdog_budget: None,
        ..watchdog_caps(50, u64::MAX)
    };
    for n in [1usize, 49, 50, 51, 237] {
        let plan = matmul_plan(&[1, 1, 8], &[8, n], Backend::SpirvVulkan, &caps);
        let x_groups = n.clamp(1, 50);
        assert_eq!(
            plan.single().expect("one dispatch").launch().grid,
            [
                (x_groups * GEMV_WIDTH) as u32,
                n.div_ceil(x_groups) as u32,
                1
            ],
            "planned grid for N={n}"
        );
        let x = fill(3, 8);
        let w = fill(4, 8 * n);
        assert_close(
            &run_plan(&plan, &x, &w, n),
            &matmul_oracle(1, 1, 8, n, &x, &w, true),
            &format!("N={n}"),
        );
    }
}

/// The batched-weight tiled GEMM (the per-expert MoE GEMM, the prefill attention matmuls) is the register-tiled
/// body with a `[K, N]` weight slab per block. Across the RADV tile ceiling the plan splits into dispatches over
/// disjoint tile ranges of the same body; the split result equals the unsplit one bit for bit and the oracle.
///
/// Mutation: in `tiled_body`'s `[K, N]` staging, drop the block's slab base (`slab`) from the weight index;
/// every block past the first reads block 0's slab and the oracle comparison goes red.
#[test]
fn batched_tiled_gemm_is_one_body_across_the_tile_ceiling() {
    let backend = Backend::SpirvVulkan;
    let unsplit_caps = DeviceCaps {
        watchdog_budget: None,
        ..DeviceCaps::wgpu_rdna3_igpu()
    };
    for (e, m, k, n) in [
        (3usize, 19usize, 24usize, 21usize),
        (2, 5, 7, 9),
        (5, 33, 17, 16),
    ] {
        let split_caps = DeviceCaps {
            known_miscompiles: poot_target::KnownMiscompiles {
                tiled_gemm_max_workgroups: Some(2),
            },
            ..unsplit_caps
        };
        let x = fill(5, e * m * k);
        let w = fill(6, e * k * n);
        let unsplit = matmul_plan(&[e, m, k], &[e, k, n], backend, &unsplit_caps);
        let split = matmul_plan(&[e, m, k], &[e, k, n], backend, &split_caps);
        assert_eq!(unsplit.dispatches().len(), 1, "E={e} {m}x{k}x{n}");
        assert!(split.dispatches().len() > 1, "E={e} {m}x{k}x{n} must split");
        for dispatch in split.dispatches().iter().chain(unsplit.dispatches()) {
            assert!(
                matches!(
                    dispatch.choice(),
                    KernelChoice::Generated(KernelRequest::Contraction(
                        ContractionSpec::DenseTiled(DenseTiledContraction {
                            weight: DenseTiledWeight::KnF32,
                            meta,
                            ..
                        })
                    )) if meta[1..3] == [n as u32, m as u32]
                ),
                "a batched-weight GEMM is the generated tiled body: {:?}",
                dispatch.choice()
            );
        }
        let got = run_plan(&split, &x, &w, e * m * n);
        let whole = run_plan(&unsplit, &x, &w, e * m * n);
        assert_eq!(
            bits(&got),
            bits(&whole),
            "E={e} {m}x{k}x{n}: split vs unsplit"
        );
        assert_close(
            &got,
            &matmul_oracle(e, m, k, n, &x, &w, false),
            &format!("E={e} {m}x{k}x{n}"),
        );
    }
}

/// The shared-weight tiled GEMM likewise: batched decode rows (`[B, 1, K] @ [K, N]` where the decode GEMV
/// declines) and a plain prefill fold into one `M` and run the one generated body, split or not.
#[test]
fn shared_weight_tiled_gemm_matches_the_oracle_across_the_tile_ceiling() {
    let backend = Backend::SpirvVulkan;
    let unsplit_caps = DeviceCaps {
        watchdog_budget: None,
        ..DeviceCaps::wgpu_rdna3_igpu()
    };
    let split_caps = DeviceCaps {
        known_miscompiles: poot_target::KnownMiscompiles {
            tiled_gemm_max_workgroups: Some(2),
        },
        ..unsplit_caps
    };
    let (m, k, n) = (37usize, 20usize, 25usize);
    let x = fill(7, m * k);
    let w = fill(8, k * n);
    let unsplit = matmul_plan(&[m, k], &[k, n], backend, &unsplit_caps);
    let split = matmul_plan(&[m, k], &[k, n], backend, &split_caps);
    assert_eq!(unsplit.dispatches().len(), 1);
    assert!(split.dispatches().len() > 1);
    let got = run_plan(&split, &x, &w, m * n);
    assert_eq!(bits(&got), bits(&run_plan(&unsplit, &x, &w, m * n)));
    assert_close(&got, &matmul_oracle(1, m, k, n, &x, &w, true), "shared");
}

/// SC-001: no contraction is an imported kernel. The planner's GEMV and GEMM routes come from `generate`, and
/// the imported family (its Rust sources, its committed assets, its manifest module) is gone.
///
/// Mutation: restore the imported family (`git checkout master -- crates/poot-graph-plan/src/imported/contraction.rs`
/// and its sources); the first assertion names the file.
#[test]
fn no_imported_contraction_kernel_exists_or_is_planned() {
    let root = poot_test_util::manifest_dir();
    let root = root.as_path();
    for gone in [
        "src/imported/contraction.rs",
        "assets/gemv_coalesced.kir.json",
        "assets/gemv_batched_coalesced.kir.json",
        "assets/tiled_gemm.kir.json",
        "assets/tiled_gemm_batched.kir.json",
        "../pootc/kernels/contraction",
    ] {
        assert!(
            !root.join(gone).exists(),
            "the imported contraction family is deleted: {gone} exists"
        );
    }
    for planner_source in ["src/planner/matmul.rs", "src/predicates.rs"] {
        let text = std::fs::read_to_string(root.join(planner_source)).unwrap();
        assert!(
            !text.contains("ImportedKernel"),
            "{planner_source} names an imported kernel: the matmul routes are generated"
        );
    }
    // Every MatMul route, on a device that takes none of the tensor-core arms, is a generated contraction.
    let caps = DeviceCaps::wgpu_rdna3_igpu();
    for (a, b) in [
        (vec![1, 1, 16], vec![16, 40]),
        (vec![3, 1, 16], vec![16, 40]),
        (vec![9, 16], vec![16, 40]),
        (vec![2, 9, 16], vec![2, 16, 40]),
        (vec![2, 4, 1, 16], vec![2, 4, 16, 40]),
    ] {
        let plan = matmul_plan(&a, &b, Backend::SpirvVulkan, &caps);
        for dispatch in plan.dispatches() {
            assert!(
                matches!(
                    dispatch.choice(),
                    KernelChoice::Generated(KernelRequest::Contraction(_))
                ),
                "{a:?} @ {b:?}: {:?}",
                dispatch.choice()
            );
        }
    }
}
