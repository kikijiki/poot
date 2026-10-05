//! Card 248, ROCm leg: measurement-only, mirrors the wgpu methodology
//! (`crates/poot-gpu/tests/graph.rs::card248_indexed_matmul_lora_tiny_r_vs_moe_scale`).
//! `lora_linear_batched`'s second `indexed_matmul` call (`xab = xa[M,r] @ B_stacked[E,r,out]`)
//! contracts over LoRA's tiny rank `r` (2-64), far below MoE's K (hundreds to thousands); this checks
//! whether the kernel has a fixed dispatch-side floor at tiny K.
//!
//! Two differences from the wgpu round:
//!
//! 1. Kernel selection. `is_indexed_gemv` (`crates/poot-graph-plan/src/predicates.rs`) gates the
//!    LDS-optimized `indexed_gemv_lds` kernel on `backend == Backend::SpirvVulkan` only. ROCm always
//!    takes the naive one-thread-per-output `indexed_matmul_dt`
//!    (`poot_kernelgen::matmul::indexed_matmul_dt`) at every K, so this measures that kernel plus
//!    ROCm's dispatch path, not an LDS-barrier floor.
//! 2. Timing method. wgpu used `GpuExecutor::new_profiled()` (timestamp-query device-time profiler)
//!    to split host-submission from device time. ROCm (Card 548: `device_time()` is always
//!    `Unknown`) has no equivalent.
//!
//! A reproducible ~1 second fixed floor per graph replay call existed on this box through the
//! pre-Card-548 `RocmGraphExecutor::run_resident_kv_recorded` path, not specific to IndexedMatMul or
//! this test (see the card 229 slope probe, since removed by card 511, which reported
//! `intercept_ms` of ~1001-1002 ms at every shape on the same box - likely GPU power-management
//! idle-state resume cost). It does not reproduce calling `Device::replay` directly (Card 548: this
//! file now measures ~80-90ms per call at the smallest shape instead), so the floor was specific to
//! that deleted path's own overhead, not `RocmContext::replay_graph_batched`.
//!
//! Method: the card 229 technique, reimplemented in this file (the original probe was removed by
//! card 511) over the executor contract's own `Device` trait (Card 548): pack `N` back-to-back
//! dispatches of the same shape into one `begin`/`dispatch`*N/`finish` recording, sweep `N`, and
//! linear-fit wall-clock of `Device::replay` alone vs `N`. The fixed per-replay floor lands in the
//! intercept and cancels from the slope.

use poot_executor::{Arg, BufferRole, Device, Dispatch};
use poot_kernel_ir::Ty;
use poot_rocm_gpu::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_target::BufferStorage;

/// Ordinary least-squares fit of `y = intercept + slope * x`, plus R^2 (the card 229 technique).
fn ols_fit(points: &[(f64, f64)]) -> (f64, f64, f64) {
    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    for &(x, y) in points {
        sxx += (x - mean_x) * (x - mean_x);
        sxy += (x - mean_x) * (y - mean_y);
    }
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let intercept = mean_y - slope * mean_x;
    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;
    for &(x, y) in points {
        let pred = intercept + slope * x;
        ss_res += (y - pred) * (y - pred);
        ss_tot += (y - mean_y) * (y - mean_y);
    }
    let r2 = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };
    (intercept, slope, r2)
}

/// For one `(m, k, n, e)` `indexed_matmul_dt` shape: build one compiled kernel, pack `n_dispatches`
/// back-to-back dispatches into a single recording (`Device::begin`/`dispatch`/`finish`), replay it
/// `repeats` times, and keep the minimum total wall-clock ms of `Device::replay` alone.
fn measure_indexed_matmul_ms_total_for_n(
    device: &mut RocmDevice,
    (m, k, n, e): (usize, usize, usize, usize),
    n_dispatches: usize,
    repeats: usize,
) -> f64 {
    assert!(n_dispatches >= 1, "n_dispatches must be >= 1");
    assert!(repeats >= 1, "repeats must be >= 1");

    // x/w content does not matter for timing, but idx must stay within [0, e) so the per-row expert
    // gather stays in-bounds.
    let x_buf = device
        .allocate(BufferRole::Input, BufferStorage::f32(), m * k)
        .expect("allocate x");
    let w_buf = device
        .allocate(BufferRole::Weight, BufferStorage::f32(), e * k * n)
        .expect("allocate w");
    let idx_buf = device
        .allocate(BufferRole::Input, BufferStorage::f32(), m)
        .expect("allocate idx");
    let idx_bytes: Vec<u8> = (0..m)
        .flat_map(|i| ((i % e) as f32).to_le_bytes())
        .collect();
    device.write(&idx_buf, &idx_bytes).expect("upload idx");
    let out_buf = device
        .allocate(BufferRole::Output, BufferStorage::f32(), m * n)
        .expect("allocate out");

    let name = format!("indexed_matmul_dt_slope_{m}x{k}x{n}x{e}");
    let body = poot_kernelgen::indexed_matmul_dt(&name, Ty::F32, m, k, n);
    let target = device.target();
    let poot_target::Backend::AmdGcn(arch) = target.backend else {
        panic!(
            "RocmDevice::target must report AmdGcn, got {:?}",
            target.backend
        );
    };
    let codegen_target = poot_codegen::Target::AmdGcn(arch);
    let tmp = poot_codegen::kernel_cache_root("rocm-card248", codegen_target);
    let out_path = poot_codegen::artifact_path(&tmp, &name, codegen_target);
    poot_codegen::compile(&body, codegen_target, &out_path).expect("compile indexed_matmul_dt");
    let bytes = std::fs::read(&out_path).expect("read compiled artifact");
    let compiled = poot_codegen::kernel_handle(&body, codegen_target, bytes);
    let kernel = device
        .load_kernel(&name, compiled)
        .expect("load indexed_matmul_dt");

    // Flat one-thread-per-output grid: `threads0 = M*N`, workgroup from the body's default.
    let wg = body.workgroup_size[0].max(1);
    let threads0 = (m * n) as u32;
    let grid_x = threads0.div_ceil(wg) * wg;

    // Untimed warmup: one begin/dispatch/finish/replay/synchronize primes the driver/HMM buffer
    // mapping before any timed replay.
    let warmup_recording = {
        device.begin(poot_graph_plan::Submission::Replay).unwrap();
        device
            .dispatch(Dispatch {
                kernel: &kernel,
                inputs: &[
                    Arg {
                        buffer: &x_buf,
                        elems: (m * k) as u32,
                    },
                    Arg {
                        buffer: &w_buf,
                        elems: (e * k * n) as u32,
                    },
                    Arg {
                        buffer: &idx_buf,
                        elems: m as u32,
                    },
                ],
                output: Arg {
                    buffer: &out_buf,
                    elems: (m * n) as u32,
                },
                threads: [grid_x, 1, 1],
                workgroup: [wg, 1, 1],
                work: (m * n) as u64,
            })
            .unwrap();
        device.finish().unwrap().expect("Replay always records")
    };
    device.replay(&warmup_recording).expect("warmup replay");
    device.synchronize().expect("warmup synchronize");

    // One recording holding `n_dispatches` identical back-to-back dispatches (same buffers; data
    // correctness does not matter, only real dims/traffic and in-bounds idx).
    device.begin(poot_graph_plan::Submission::Replay).unwrap();
    for _ in 0..n_dispatches {
        device
            .dispatch(Dispatch {
                kernel: &kernel,
                inputs: &[
                    Arg {
                        buffer: &x_buf,
                        elems: (m * k) as u32,
                    },
                    Arg {
                        buffer: &w_buf,
                        elems: (e * k * n) as u32,
                    },
                    Arg {
                        buffer: &idx_buf,
                        elems: m as u32,
                    },
                ],
                output: Arg {
                    buffer: &out_buf,
                    elems: (m * n) as u32,
                },
                threads: [grid_x, 1, 1],
                workgroup: [wg, 1, 1],
                work: (m * n) as u64,
            })
            .unwrap();
    }
    let recording = device.finish().unwrap().expect("Replay always records");

    // Replay the same recording `repeats` times, keep the minimum total wall-clock; only
    // `Device::replay` is inside the `Instant` window.
    let mut min_ms = f64::INFINITY;
    for _ in 0..repeats {
        let t0 = std::time::Instant::now();
        device.replay(&recording).expect("replay");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if ms < min_ms {
            min_ms = ms;
        }
    }
    min_ms
}

/// OLS slope fit over `n_sweep` for one `(m, k, n, e)` shape: returns `(slope_ms_per_dispatch,
/// intercept_ms, r2)`.
fn fit_indexed_matmul_slope(
    device: &mut RocmDevice,
    shape: (usize, usize, usize, usize),
    n_sweep: &[usize],
    repeats: usize,
) -> (f64, f64, f64, Vec<(usize, f64)>) {
    let mut per_n_raw = Vec::with_capacity(n_sweep.len());
    for &n_dispatches in n_sweep {
        let total_ms = measure_indexed_matmul_ms_total_for_n(device, shape, n_dispatches, repeats);
        per_n_raw.push((n_dispatches, total_ms));
    }
    let points: Vec<(f64, f64)> = per_n_raw.iter().map(|&(n, ms)| (n as f64, ms)).collect();
    let (intercept_ms, slope_ms_per_dispatch, r2) = ols_fit(&points);
    (slope_ms_per_dispatch, intercept_ms, r2, per_n_raw)
}

#[test]
#[ignore = "perf measurement (card 248, ROCm leg): naive single-call-per-iteration wall-clock \
            calibration, kept as a quick (~5s) reproduction of this box's real ~1s \
            replay floor (see module doc comment) - NOT the real per-dispatch \
            measurement (see card248_indexed_matmul_rocm_k_sweep_slope for that). Run with \
            --ignored --nocapture --test-threads=1."]
fn card248_indexed_matmul_rocm_naive_percall_floor_calibration() {
    use std::time::Instant;

    let Some(mut device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())
    else {
        return;
    };
    let (m, k, n, e) = (8usize, 8usize, 896usize, 8usize);

    eprintln!(
        "\n[card248 ROCm] naive single-call-per-iteration calibration at K=8 (LoRA-typical), M=8, \
         N=896, E=8 - 5 consecutive one-dispatch replay calls, printed individually to rule out \
         a one-time cold-start cost"
    );
    for call in 0..5 {
        let t = Instant::now();
        let _ = measure_indexed_matmul_ms_total_for_n(&mut device, (m, k, n, e), 1, 1);
        let us = t.elapsed().as_secs_f64() * 1e6;
        eprintln!("  call {call}: {us:.1} us");
    }
    eprintln!(
        "  => if every call above lands near ~1.0-1.1 SECOND (not microseconds), that is this box's \
         pre-Card-548 ~1s-per-replay-call floor (see module doc comment); measured post-Card-548 via \
         Device::replay directly, this floor did not reproduce (~80-90ms per call instead) - likely \
         specific to the deleted RocmGraphExecutor::run_resident_kv_recorded path, not the raw \
         RocmContext::replay_graph_batched primitive this now calls through. See \
         card248_indexed_matmul_rocm_k_sweep_slope for the real per-dispatch numbers."
    );
}

#[test]
#[ignore = "perf measurement (card 248, ROCm leg): OLS slope-fit K-sweep for indexed_matmul_dt, \
            isolating real per-dispatch device time from this box's ~1s fixed per-replay floor \
            (see module doc comment) using the card-229 slope-fit technique reimplemented in this file. \
            Run with --ignored --nocapture --test-threads=1. Check ps aux/free -h for GPU \
            contention first - this box allows only ONE active HSA queue system-wide."]
fn card248_indexed_matmul_rocm_k_sweep_slope() {
    let Some(mut device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())
    else {
        return;
    };

    eprintln!(
        "\n[card248 ROCm] K-sweep (M=8, N=896 qwen2.5-0.5b hidden, E=8), OLS slope fit over \
         n_sweep=[1,2,4,8,16,32] repeats=5 - ROCm always uses the naive indexed_matmul_dt kernel \
         (is_indexed_gemv gates SpirvVulkan only), so this measures the naive kernel's own \
         K-scaling, not an LDS-barrier floor like wgpu's indexed_gemv_lds"
    );
    let (m, n, e) = (8usize, 896usize, 8usize);
    let n_sweep = [1usize, 2, 4, 8, 16, 32];
    let repeats = 5usize;
    let ks: [(&str, usize); 10] = [
        ("r=2", 2),
        ("r=4", 4),
        ("r=8", 8),
        ("r=16", 16),
        ("r=32", 32),
        ("r=64", 64),
        ("r=128", 128),
        ("r=256", 256),
        ("K=896 (qwen2.5-0.5b hidden, dense-equiv)", 896),
        ("K=1024 (mixtral-tiny hidden, MoE-scale)", 1024),
    ];
    let mut rows: Vec<(usize, f64, f64, f64)> = Vec::new(); // (k, slope_ms, intercept_ms, r2)
    for (label, k) in ks {
        let (slope_ms, intercept_ms, r2, per_n_raw) =
            fit_indexed_matmul_slope(&mut device, (m, k, n, e), &n_sweep, repeats);
        let ops = 2.0 * m as f64 * k as f64 * n as f64;
        let gflops = ops / (slope_ms * 1e-3) / 1e9;
        let raw_str = per_n_raw
            .iter()
            .map(|&(nd, ms)| format!("n={nd}:{ms:.4}ms"))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "  K={k:>5} ({label:<42}) slope={slope_ms:>9.5} ms/dispatch  GFLOPS={gflops:>8.3}  \
             intercept={intercept_ms:>9.4} ms  r2={r2:.4}  raw=[{raw_str}]"
        );
        rows.push((k, slope_ms, intercept_ms, r2));
    }
    let tiny = rows.iter().find(|&&(k, ..)| k == 8).unwrap();
    let moe = rows.iter().find(|&&(k, ..)| k == 1024).unwrap();
    let tiny_gflops = 2.0 * m as f64 * 8.0 * n as f64 / (tiny.1 * 1e-3) / 1e9;
    let moe_gflops = 2.0 * m as f64 * 1024.0 * n as f64 / (moe.1 * 1e-3) / 1e9;
    eprintln!(
        "  => r=8 (LoRA-typical) slope {:.5} ms/dispatch vs K=1024 (MoE-scale) slope {:.5} \
         ms/dispatch: {:.2}x device time for {:.0}x fewer FLOPs (K ratio) => {:.1}x lower GFLOPS/s \
         at tiny r",
        tiny.1,
        moe.1,
        tiny.1 / moe.1,
        1024.0 / 8.0,
        moe_gflops / tiny_gflops
    );
    let expected_linear_ms = moe.1 * (8.0 / 1024.0);
    eprintln!(
        "  => pure-FLOP-linear scaling from K=1024's slope {:.5} ms would predict K=8's slope at \
         {:.5} ms (moe_slope * 8/1024) - measured {:.5} ms is {:.2}x that prediction",
        moe.1,
        expected_linear_ms,
        tiny.1,
        tiny.1 / expected_linear_ms
    );
    eprintln!(
        "  note: every intercept above should be roughly the SAME ~1000 ms regardless of K (the \
         fixed per-replay floor this module's doc comment describes) - if it instead scales with K, \
         that would mean the floor is not purely fixed and this fit's slope/intercept split needs a \
         second look before trusting the slope numbers."
    );
}
