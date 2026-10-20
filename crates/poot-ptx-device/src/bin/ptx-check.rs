//! PTX execution check: load poot-emitted `.ptx` files (generated locally by poot-codegen via `llc`) and
//! run them on an NVIDIA GPU, verifying the results against the obvious CPU computation. This is the
//! NVPTX-path executor-equivalence check (oracle stage 2), run on a rented GPU since the dev box has no
//! NVIDIA card.
//!
//! It dispatches through `poot_ptx_runtime::PtxContext`, so it doubles as the runtime's integration test:
//! upload, dispatch (the `(ptr, i64 len)` ABI) and download.
//!
//! Usage: `ptx-check <DIR>` where DIR holds `add.ptx`, `square.ptx`, `matmul.ptx`, ...

use std::cell::{Cell, RefCell};

use poot_ptx_runtime::PtxContext;
use poot_runtime_common::{ArgAccess, ArgSchema, CompiledKernel, DeviceBackend, KernelCode};
use poot_target::{Backend, ElementKind, TargetId};

/// Wrap a poot-codegen-emitted `.ptx` file's text as a dispatchable [`CompiledKernel`] (card 608's
/// sanctioned "imported kernel" escape hatch: this binary loads compiled `.ptx` text straight off disk,
/// not through `poot_codegen::kernel_handle`, so it builds the handle itself). `elements` is one entry
/// per data buffer in binding order, inputs first, the single output (`ArgAccess::Write`) last.
fn ptx_kernel(ptx: &str, entry: &str, elements: &[ElementKind]) -> CompiledKernel {
    let args: Vec<ArgSchema> = elements
        .iter()
        .enumerate()
        .map(|(i, &element)| {
            let access = if i + 1 == elements.len() {
                ArgAccess::Write
            } else {
                ArgAccess::Read
            };
            ArgSchema::new(element, access)
        })
        .collect();
    // SAFETY: `ptx` is poot-codegen's `Target::Nvptx` output for `entry` (this binary's usage doc), whose
    // parameters are the caller-declared `elements` in binding order.
    unsafe {
        CompiledKernel::new(
            TargetId::Gpu(Backend::Nvptx),
            entry,
            KernelCode::Ptx(ptx.into()),
            args,
            false,
        )
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| ".".to_string());

    // Capture `c = a + b`, replay it, then update `b` in place and replay again: a captured graph must
    // replay with one host call and re-read updated buffer contents at the recorded (stable) address.
    if args.iter().any(|a| a == "--graph-probe") {
        std::process::exit(graph_probe(&dir));
    }

    let ctx = PtxContext::new().expect("init PtxContext (libcuda present?)");
    let fixtures = Fixtures::new(&dir);
    let mut failures = 0;

    // Base fixtures (add/square/matmul/exp/silu/reduce/bcast/gather/dus): each is individually
    // staged-gated (`Fixtures`) so a partial PTX dir prints a SKIP line instead of panicking; a staged but
    // wrong fixture still runs and can still fail the check.
    // add: c = a + b
    if fixtures.staged("add") {
        {
            let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
            let n = a.len();
            let c = run(
                &ctx,
                &dir,
                "add",
                "add",
                &[&a, &b],
                n,
                [64, 1, 1],
                [n as u32, 1, 1],
            );
            let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
            report("add", &c, &want, &mut failures);
        }
    }
    // square: y = x * x
    if fixtures.staged("square") {
        {
            let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let n = x.len();
            let y = run(
                &ctx,
                &dir,
                "square",
                "square",
                &[&x],
                n,
                [64, 1, 1],
                [n as u32, 1, 1],
            );
            let want: Vec<f32> = x.iter().map(|v| v * v).collect();
            report("square", &y, &want, &mut failures);
        }
    }
    // matmul: C[2,3] = A[2,4] @ B[4,3] (2D grid: x = N cols, y = M rows; 16x16 block).
    if fixtures.staged("matmul") {
        {
            let (m, n, k) = (2usize, 3, 4);
            let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();
            let c = run(
                &ctx,
                &dir,
                "matmul",
                "matmul",
                &[&a, &b],
                m * n,
                [16, 16, 1],
                [n as u32, m as u32, 1],
            );
            let mut want = vec![0.0f32; m * n];
            for r in 0..m {
                for col in 0..n {
                    let mut acc = 0.0;
                    for kk in 0..k {
                        acc += a[r * k + kk] * b[kk * n + col];
                    }
                    want[r * n + col] = acc;
                }
            }
            report("matmul", &c, &want, &mut failures);
        }
    }

    // generated kernels (from kernelgen): exercise the NVPTX emitter paths.
    if fixtures.staged("exp") {
        let x: Vec<f32> = vec![0.0, 1.0, -1.0, 2.0];
        let e = run(&ctx, &dir, "exp", "exp", &[&x], 4, [64, 1, 1], [4, 1, 1]);
        report_close(
            "exp",
            &e,
            &x.iter().map(|v| v.exp()).collect::<Vec<_>>(),
            2e-3,
            &mut failures,
        );
    }
    if fixtures.staged("silu") {
        let x: Vec<f32> = vec![0.0, 1.0, -1.0, 2.0];
        let s = run(&ctx, &dir, "silu", "silu", &[&x], 4, [64, 1, 1], [4, 1, 1]);
        report_close(
            "silu",
            &s,
            &x.iter().map(|v| v / (1.0 + (-v).exp())).collect::<Vec<_>>(),
            2e-3,
            &mut failures,
        );
    }
    if fixtures.staged("reduce") {
        // reduce_last sum over cols=4, 2 rows.
        let xr: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let r = run(
            &ctx,
            &dir,
            "reduce",
            "reduce",
            &[&xr],
            2,
            [64, 1, 1],
            [2, 1, 1],
        );
        report("reduce", &r, &[10.0, 100.0], &mut failures);
    }
    // broadcast [3] -> [2,3]
    if fixtures.staged("bcast") {
        let bias: Vec<f32> = vec![10.0, 20.0, 30.0];
        let b = run(
            &ctx,
            &dir,
            "bcast",
            "bcast",
            &[&bias],
            6,
            [64, 1, 1],
            [6, 1, 1],
        );
        report(
            "bcast",
            &b,
            &[10.0, 20.0, 30.0, 10.0, 20.0, 30.0],
            &mut failures,
        );
    }
    // gather_axis0: data[3,2], index[2,0,1]
    if fixtures.staged("gather") {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let index: Vec<f32> = vec![2.0, 0.0, 1.0];
        let g = run(
            &ctx,
            &dir,
            "gather",
            "gather",
            &[&data, &index],
            6,
            [64, 1, 1],
            [6, 1, 1],
        );
        report("gather", &g, &[5.0, 6.0, 1.0, 2.0, 3.0, 4.0], &mut failures);
    }
    // DynamicUpdateSlice (KV slot write): operand[2,4,3]=0..24, write update[2,1,3] at slot idx=2
    // on axis 1; slot 2 (flat 6..9 and 18..21) is overwritten, the rest preserved.
    if fixtures.staged("dus") {
        let operand: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let update: Vec<f32> = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0];
        let dus = run(
            &ctx,
            &dir,
            "dus",
            "dus",
            &[&operand, &update],
            24,
            [64, 1, 1],
            [24, 1, 1],
        );
        let mut want_dus: Vec<f32> = (0..24).map(|i| i as f32).collect();
        want_dus[6..9].copy_from_slice(&[100.0, 200.0, 300.0]);
        want_dus[18..21].copy_from_slice(&[400.0, 500.0, 600.0]);
        report("dus", &dus, &want_dus, &mut failures);
    }

    // Kernels authored in ordinary Rust and compiled by pootc (Stable MIR -> Body -> PTX), if staged.
    // Entry names are the mangled `__poot_kernel_*`.
    if fixtures.staged("imported_add") {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let c = run(
            &ctx,
            &dir,
            "imported_add",
            "__poot_kernel_add",
            &[&a, &b],
            5,
            [64, 1, 1],
            [5, 1, 1],
        );
        report(
            "imported_add",
            &c,
            &[11.0, 22.0, 33.0, 44.0, 55.0],
            &mut failures,
        );

        // imported matmul: C[2,3] = A[2,4] @ B[4,3] (dims baked in; 1D grid over m*n outputs).
        let (m, n, k) = (2usize, 3, 4);
        let am: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let bm: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();
        let cm = run(
            &ctx,
            &dir,
            "imported_matmul",
            "__poot_kernel_matmul",
            &[&am, &bm],
            m * n,
            [64, 1, 1],
            [(m * n) as u32, 1, 1],
        );
        let mut wantm = vec![0.0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0.0;
                for kk in 0..k {
                    acc += am[r * k + kk] * bm[kk * n + col];
                }
                wantm[r * n + col] = acc;
            }
        }
        report("imported_matmul", &cm, &wantm, &mut failures);
    }

    // Imported kernels on the PTX path: the decode GEMV (+ bias), the on-device argmax, and the coarsened
    // tiled GEMM. Shapes mirror the wgpu import_run tests. Each block runs only if its .ptx is staged.
    if fixtures.staged("gemv") {
        // decode GEMV: out[col] = sum_k x[k] * w[k*N+col]. 128 lanes/output (one block per output column).
        let (k, n) = (4usize, 8usize);
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1 - 0.2).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let got = run(
            &ctx,
            &dir,
            "gemv",
            "__poot_kernel_gemv",
            &[&x, &w],
            n,
            [128, 1, 1],
            [(n * 128) as u32, 1, 1],
        );
        let want: Vec<f32> = (0..n)
            .map(|col| (0..k).map(|kk| x[kk] * w[kk * n + col]).sum())
            .collect();
        report_close("imported_gemv", &got, &want, 1e-5, &mut failures);
    }
    if fixtures.staged("gemv_bias") {
        let (k, n) = (4usize, 8usize);
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1 - 0.2).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2 - 0.5).collect();
        let got = run(
            &ctx,
            &dir,
            "gemv_bias",
            "__poot_kernel_gemv_bias",
            &[&x, &w, &bias],
            n,
            [128, 1, 1],
            [(n * 128) as u32, 1, 1],
        );
        let want: Vec<f32> = (0..n)
            .map(|col| (0..k).map(|kk| x[kk] * w[kk * n + col]).sum::<f32>() + bias[col])
            .collect();
        report_close("imported_gemv_bias", &got, &want, 1e-5, &mut failures);
    }
    if fixtures.staged("argmax") {
        // on-device greedy argmax: 64 lanes scan the logits, lane 0 writes the winning index into out[0].
        let vocab = 64usize;
        let mut logits: Vec<f32> = (0..vocab).map(|i| (i as f32) * 0.01).collect();
        logits[42] = 99.0; // the clear max
        let got = run(
            &ctx,
            &dir,
            "argmax",
            "__poot_kernel_argmax",
            &[&logits],
            1,
            [64, 1, 1],
            [64, 1, 1],
        );
        report("imported_argmax", &got, &[42.0], &mut failures);
    }
    if fixtures.staged("tiled_gemm") {
        // coarsened tiled GEMM: C[M,N] = A[M,K] @ B[K,N], dims = [M,K,N,offset]. One TSxTS tile per block,
        // 64 lanes; grid = ceil(M/16)*ceil(N/8) blocks. M=16,K=8,N=8 -> 1 tile.
        let (m, k, n) = (16usize, 8usize, 8usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.7).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.07 - 0.3).collect();
        let dims = [m as u32, k as u32, n as u32, 0u32];
        let tiles = m.div_ceil(16) * n.div_ceil(8);
        let got = run_dims(
            &ctx,
            &dir,
            "tiled_gemm",
            "__poot_kernel_tiled_gemm_coarsened",
            &[&a, &b],
            &dims,
            m * n,
            [64, 1, 1],
            [(tiles * 64) as u32, 1, 1],
        );
        let want: Vec<f32> = (0..m * n)
            .map(|idx| {
                let (r, c) = (idx / n, idx % n);
                (0..k).map(|j| a[r * k + j] * b[j * n + c]).sum()
            })
            .collect();
        report_close("imported_tiled_gemm", &got, &want, 1e-4, &mut failures);
    }

    // Attention decode: the generated online-softmax kernel, 4 lanes per head. Dims match the staged kernel
    // (Hq=4, n_rep=2, 6 keys, D=8); compared to the textbook two-pass attention.
    if fixtures.staged("attention_decode") {
        let (hq, n_rep, cap, d, lanes) = (4usize, 2usize, 6usize, 8usize, 4usize);
        let hkv = hq / n_rep;
        let scale = 0.35355338f32;
        let q = det_fill(hq * d, 1);
        let k = det_fill(hkv * cap * d, 2);
        let v = det_fill(hkv * cap * d, 3);
        let mask: Vec<f32> = (0..cap)
            .map(|t| if t <= 3 { 0.0 } else { -1.0e9 })
            .collect();
        let got = run(
            &ctx,
            &dir,
            "attention_decode",
            "attention_decode",
            &[&q, &k, &v, &mask],
            hq * d,
            [lanes as u32, 1, 1],
            [(hq * lanes) as u32, 1, 1],
        );
        let want = attention_decode_ref(&q, &k, &v, &mask, hq, n_rep, cap, d, scale);
        report_close("attention_decode", &got, &want, 1e-4, &mut failures);
    }

    match fixtures.verdict(
        failures,
        DeviceBackend::Ptx.required(|name| std::env::var(name).ok()),
    ) {
        Ok(summary) => println!("{summary}"),
        Err(reason) => {
            eprintln!("{reason}");
            std::process::exit(1);
        }
    }
}

/// Load `<dir>/<file>.ptx`, dispatch `entry` through the PtxContext with read-only `inputs` (uploaded)
/// and one writable output of `out_len` (allocated), and return the output. `label` names it for the
/// profiler + the per-kernel device-time line.
#[expect(clippy::too_many_arguments)]
fn run(
    ctx: &PtxContext,
    dir: &str,
    label: &str,
    entry: &str,
    inputs: &[&[f32]],
    out_len: usize,
    block: [u32; 3],
    threads: [u32; 3],
) -> Vec<f32> {
    let path = format!("{dir}/{label}.ptx");
    let ptx = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let in_bufs: Vec<_> = inputs
        .iter()
        .map(|d| ctx.upload_f32(d).expect("upload"))
        .collect();
    let in_refs: Vec<_> = in_bufs.iter().collect();
    let in_lens: Vec<u32> = in_bufs.iter().map(|b| b.elem_count()).collect();
    let out = ctx.alloc_f32(out_len).expect("alloc out");
    let elements: Vec<ElementKind> = (0..=inputs.len()).map(|_| ElementKind::F32).collect();
    let kernel = ptx_kernel(&ptx, entry, &elements);
    ctx.dispatch_dev(
        label,
        &kernel,
        block,
        threads,
        &in_refs,
        &in_lens,
        &out,
        out.elem_count(),
    )
    .expect("dispatch");
    ctx.download_f32(&out).expect("download")
}

/// Like [`run`], but binds an extra `u32` metadata buffer (e.g. a tiled GEMM's `dims = [M,K,N,offset]`)
/// between the f32 data inputs and the output - the imported shape-generic kernels read their shape from it.
/// Uploaded via `upload_i32` (the bytes are identical for the small non-negative dims).
#[expect(clippy::too_many_arguments)]
fn run_dims(
    ctx: &PtxContext,
    dir: &str,
    label: &str,
    entry: &str,
    f32_inputs: &[&[f32]],
    dims: &[u32],
    out_len: usize,
    block: [u32; 3],
    threads: [u32; 3],
) -> Vec<f32> {
    let path = format!("{dir}/{label}.ptx");
    let ptx = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut in_bufs: Vec<_> = f32_inputs
        .iter()
        .map(|d| ctx.upload_f32(d).expect("upload f32"))
        .collect();
    let dims_i32: Vec<i32> = dims.iter().map(|&x| x as i32).collect();
    in_bufs.push(ctx.upload_i32(&dims_i32).expect("upload dims"));
    let in_refs: Vec<_> = in_bufs.iter().collect();
    let in_lens: Vec<u32> = in_bufs.iter().map(|b| b.elem_count()).collect();
    let out = ctx.alloc_f32(out_len).expect("alloc out");
    let mut elements: Vec<ElementKind> = f32_inputs.iter().map(|_| ElementKind::F32).collect();
    elements.push(ElementKind::I32); // dims
    elements.push(ElementKind::F32); // out
    let kernel = ptx_kernel(&ptx, entry, &elements);
    ctx.dispatch_dev(
        label,
        &kernel,
        block,
        threads,
        &in_refs,
        &in_lens,
        &out,
        out.elem_count(),
    )
    .expect("dispatch");
    ctx.download_f32(&out).expect("download")
}

/// Deterministic pseudo-random fill in [-1, 1) (xorshift64), matching `poot-kernelgen`'s `tests/flash_interp.rs::fill`.
fn det_fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
        })
        .collect()
}

/// The textbook two-pass decode attention: `softmax(scale * q.k + mask) @ v` per head, GQA `kv = h / n_rep`.
#[expect(clippy::too_many_arguments)]
fn attention_decode_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
    hq: usize,
    n_rep: usize,
    cap: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; hq * d];
    for h in 0..hq {
        let kv = h / n_rep;
        let scores: Vec<f32> = (0..cap)
            .map(|t| {
                let dot: f32 = (0..d)
                    .map(|c| q[h * d + c] * k[kv * cap * d + t * d + c])
                    .sum();
                dot * scale + mask[t]
            })
            .collect();
        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
        let denominator: f32 = weights.iter().sum();
        for c in 0..d {
            let acc: f32 = (0..cap)
                .map(|t| weights[t] * v[kv * cap * d + t * d + c])
                .sum();
            out[h * d + c] = acc / denominator;
        }
    }
    out
}

fn report_close(name: &str, got: &[f32], want: &[f32], tol: f32, failures: &mut u32) {
    let ok = got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| (g - w).abs() <= tol * w.abs().max(1.0));
    if ok {
        println!("  {name}: PASS  {got:?}");
    } else {
        *failures += 1;
        eprintln!("  {name}: FAIL  got {got:?} want {want:?}");
    }
}

fn report(name: &str, got: &[f32], want: &[f32], failures: &mut u32) {
    if got == want {
        println!("  {name}: PASS  {got:?}");
    } else {
        *failures += 1;
        eprintln!("  {name}: FAIL  got {got:?} want {want:?}");
    }
}

/// Capture/replay probe. Returns a process exit code (0 = pass). Uses a non-profiled context so the
/// captured dispatch does not sync the stream mid-capture.
fn graph_probe(dir: &str) -> i32 {
    let ctx = PtxContext::new().expect("init PtxContext (libcuda present?)");
    let path = format!("{dir}/add.ptx");
    let ptx = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));

    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let da = ctx.upload_f32(&a).expect("upload a");
    let db = ctx.upload_f32(&b).expect("upload b");
    let dc = ctx.alloc_f32(a.len()).expect("alloc c");

    let kernel = ptx_kernel(
        &ptx,
        "add",
        &[ElementKind::F32, ElementKind::F32, ElementKind::F32],
    );

    // warm up outside capture: the first dispatch loads + JITs the module (cuModuleLoadData), which must
    // not happen mid-capture.
    ctx.dispatch_dev(
        "add",
        &kernel,
        [64, 1, 1],
        [a.len() as u32, 1, 1],
        &[&da, &db],
        &[da.elem_count(), db.elem_count()],
        &dc,
        dc.elem_count(),
    )
    .expect("warmup dispatch");
    ctx.synchronize().expect("warmup sync");

    // capture one dispatch (c = a + b) into a replayable graph.
    ctx.begin_capture().expect("begin_capture");
    ctx.dispatch_dev(
        "add",
        &kernel,
        [64, 1, 1],
        [a.len() as u32, 1, 1],
        &[&da, &db],
        &[da.elem_count(), db.elem_count()],
        &dc,
        dc.elem_count(),
    )
    .expect("dispatch during capture");
    let exec = ctx.end_capture().expect("end_capture");

    let mut failures = 0u32;
    // replay #1: should compute a + b.
    exec.launch().expect("launch 1");
    ctx.synchronize().expect("sync 1");
    let c1 = ctx.download_f32(&dc).expect("download 1");
    report(
        "graph replay (a+b)",
        &c1,
        &[11.0, 22.0, 33.0, 44.0, 55.0],
        &mut failures,
    );

    // update b in place at its stable address, then replay the same graph: must re-read new b.
    let b2 = [100.0f32, 200.0, 300.0, 400.0, 500.0];
    ctx.update_f32(&db, &b2).expect("update b");
    exec.launch().expect("launch 2");
    ctx.synchronize().expect("sync 2");
    let c2 = ctx.download_f32(&dc).expect("download 2");
    report(
        "graph replay after in-place update (a+b2)",
        &c2,
        &[101.0, 202.0, 303.0, 404.0, 505.0],
        &mut failures,
    );

    if failures == 0 {
        println!("PTX GRAPH PROBE PASSED (capture + replay + stable-buffer in-place update)");
        0
    } else {
        eprintln!("{failures} PTX GRAPH PROBE CHECK(S) FAILED");
        1
    }
}

/// The `.ptx` fixtures one run looks for in its directory: which were staged, which were not.
///
/// A check that ran nothing is not a pass: [`Fixtures::verdict`] fails a run that found no staged fixture, and
/// fails one that lacks any fixture when the PTX backend is required (`POOT_REQUIRE_PTX=1`, set by `just
/// test-device-ptx`). Without the requirement a partial directory checks what is present and reports the rest as
/// skipped, as a developer staging one kernel wants.
struct Fixtures<'a> {
    dir: &'a str,
    staged: Cell<u32>,
    missing: RefCell<Vec<String>>,
}

impl<'a> Fixtures<'a> {
    fn new(dir: &'a str) -> Self {
        Self {
            dir,
            staged: Cell::new(0),
            missing: RefCell::new(Vec::new()),
        }
    }

    /// True when `{dir}/{label}.ptx` is staged. A missing one prints a SKIP line and is remembered for
    /// [`Self::verdict`]. A staged fixture runs and can still fail the check.
    fn staged(&self, label: &str) -> bool {
        if std::path::Path::new(&format!("{}/{label}.ptx", self.dir)).exists() {
            self.staged.set(self.staged.get() + 1);
            true
        } else {
            println!("  {label}: SKIP (not staged)");
            self.missing.borrow_mut().push(label.to_string());
            false
        }
    }

    /// The run's outcome once every fixture was looked for: `Ok` with the summary line, or `Err` with the reason it
    /// is not a pass.
    fn verdict(&self, failures: u32, required: bool) -> Result<String, String> {
        let staged = self.staged.get();
        let missing = self.missing.borrow();
        if failures > 0 {
            return Err(format!("{failures} PTX CHECK(S) FAILED"));
        }
        if staged == 0 {
            return Err(format!(
                "no fixture staged in {}: nothing was checked ({} looked for)",
                self.dir,
                missing.len()
            ));
        }
        if required && !missing.is_empty() {
            return Err(format!(
                "POOT_REQUIRE_PTX=1 and {} fixture(s) not staged in {}: {}",
                missing.len(),
                self.dir,
                missing.join(", ")
            ));
        }
        Ok(format!(
            "ALL PTX CHECKS PASSED ({staged} fixtures run, {} skipped)",
            missing.len()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::Fixtures;

    fn stage(name: &str, files: &[&str]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ptx-check-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for file in files {
            std::fs::write(dir.join(format!("{file}.ptx")), "// fake ptx for gate test").unwrap();
        }
        dir
    }

    #[test]
    fn missing_fixture_is_skipped_not_run() {
        let dir = stage("missing", &[]);
        let fixtures = Fixtures::new(dir.to_str().unwrap());
        assert!(!fixtures.staged("add"));
        assert!(!fixtures.staged("matmul"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn staged_fixture_is_run_not_skipped() {
        let dir = stage("present", &["add"]);
        let fixtures = Fixtures::new(dir.to_str().unwrap());
        assert!(fixtures.staged("add"));
        assert!(!fixtures.staged("square"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A directory with no fixture is a failed check, required or not (SC-003 of card 1218).
    #[test]
    fn a_directory_with_no_fixture_is_not_a_pass() {
        let dir = stage("empty", &[]);
        let fixtures = Fixtures::new(dir.to_str().unwrap());
        assert!(!fixtures.staged("add"));
        for required in [false, true] {
            let error = fixtures.verdict(0, required).unwrap_err();
            assert!(error.contains("no fixture staged"), "{error}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A partial directory passes unless the backend is required, in which case the missing fixtures fail it.
    #[test]
    fn a_partial_directory_fails_only_when_ptx_is_required() {
        let dir = stage("partial", &["add"]);
        let fixtures = Fixtures::new(dir.to_str().unwrap());
        assert!(fixtures.staged("add"));
        assert!(!fixtures.staged("square"));
        assert!(
            fixtures
                .verdict(0, false)
                .unwrap()
                .contains("1 fixtures run, 1 skipped")
        );
        let error = fixtures.verdict(0, true).unwrap_err();
        assert!(
            error.contains("POOT_REQUIRE_PTX=1") && error.contains("square"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_comparison_fails_the_run() {
        let dir = stage("failed", &["add"]);
        let fixtures = Fixtures::new(dir.to_str().unwrap());
        assert!(fixtures.staged("add"));
        assert_eq!(
            fixtures.verdict(2, false).unwrap_err(),
            "2 PTX CHECK(S) FAILED"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
