//! Card 530 (owner PTX is first class, not compile-only-verified): real NVIDIA dispatch of
//! `wmma_tile` (the target-neutral single 16x16x16 fragment tile, F16 operands / F32 accumulate), compiled
//! to real PTX (`poot_codegen::compile`, `Target::Nvptx`) and run via `poot_ptx_runtime::PtxContext`'s
//! `dispatch_dev`. The NVPTX sibling of `poot-kernelgen`'s `coopmat_dispatch_probe` (wgpu/RADV coopmat) and
//! `poot-rocm-gpu`'s `tests::resident::probe_*_wmma_rocm` (ROCm WMMA): all three exercise the same fragment
//! op lowered to a different target and compare against the same CPU oracle.
//!
//! `PtxContext::new()` already implements the project's device-skip convention (AGENTS.md): it returns
//! `Err` when no NVIDIA driver/device is present, and panics instead when `POOT_REQUIRE_PTX=1` (set by
//! `just test-device-ptx` and the PTX pod runs), so this test never reports a skip as a
//! pass on a lane that was supposed to run it.

use poot_codegen::{Target, artifact_path, compile};
use poot_kernelgen as kg;
use poot_ptx_runtime::PtxContext;

/// Encode f32 -> f16 bytes (little-endian), reusing the loader's own f32->f16 narrowing
/// (`poot_load::gguf::f32_to_f16`), matching `coopmat_dispatch_probe.rs`'s helper.
fn f16_encode(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|&f| poot_load::gguf::f32_to_f16(f).to_le_bytes())
        .collect()
}

#[test]
fn wmma_tile_dispatches_and_matches_cpu_on_ptx() {
    let ctx = match PtxContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("SKIP wmma_tile_dispatches_and_matches_cpu_on_ptx: no PTX device ({e})");
            return;
        }
    };

    let body = poot_test_util::kernel_fixtures::wmma_tile("k");
    let dir = std::env::temp_dir().join("poot-ptx-wmma-dispatch-probe");
    std::fs::create_dir_all(&dir).unwrap();
    let out = artifact_path(&dir, "wmma_tile_ptx", Target::Nvptx);
    compile(&body, Target::Nvptx, &out).expect("wmma_tile must compile to PTX");
    let ptx_bytes = std::fs::read(&out).expect("read compiled PTX");
    let kernel = poot_codegen::kernel_handle(&body, Target::Nvptx, ptx_bytes);

    let a_data: Vec<f32> = (0..256).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let b_data: Vec<f32> = (0..256).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
    let a_buf = ctx
        .upload_f16(&f16_encode(&a_data))
        .expect("upload A fragment tile");
    let b_buf = ctx
        .upload_f16(&f16_encode(&b_data))
        .expect("upload B fragment tile");
    let out_buf = ctx.alloc_f32(256).expect("allocate C fragment tile");

    ctx.dispatch_dev(
        "wmma_tile_ptx",
        &kernel,
        body.workgroup_size,
        body.workgroup_size, // one warp, one tile: threads == block
        &[&a_buf, &b_buf],
        &[a_buf.elem_count(), b_buf.elem_count()],
        &out_buf,
        out_buf.elem_count(),
    )
    .expect("real PTX WMMA dispatch");
    let got = ctx
        .download_f32(&out_buf)
        .expect("download C fragment tile");

    // CPU oracle: round A/B to f16 (the WMMA load narrows on the way in), f32-accumulate, k=0..16 -
    // matching the RADV coopmat sibling probe's oracle shape (`coopmat_dispatch_probe.rs`).
    let round = |x: f32| poot_quant::scalar::f16_to_f32(poot_load::gguf::f32_to_f16(x));
    let ra: Vec<f32> = a_data.iter().map(|&x| round(x)).collect();
    let rb: Vec<f32> = b_data.iter().map(|&x| round(x)).collect();
    let mut max_abs = 0.0f32;
    let mut first_bad = None;
    for i in 0..16 {
        for j in 0..16 {
            let mut acc = 0.0f32;
            for kk in 0..16 {
                acc += ra[i * 16 + kk] * rb[kk * 16 + j];
            }
            let g = got[i * 16 + j];
            let d = (g - acc).abs();
            max_abs = max_abs.max(d);
            let tol = acc.abs() * 0.02 + 1e-2;
            // `d > tol` is false for a NaN `d`, so a NaN result would pass without the explicit check.
            if (d.is_nan() || d > tol) && first_bad.is_none() {
                first_bad = Some((i, j, acc, g));
            }
        }
    }
    if let Some((i, j, want, g)) = first_bad {
        panic!(
            "wmma_tile PTX MISMATCH at [{i},{j}]: want {want:.4} got {g:.4} (max_abs={max_abs:.3e})"
        );
    }
    eprintln!("wmma_tile [16x16]@[16x16] OK on real PTX WMMA (max_abs={max_abs:.3e})");
}

/// card 674: `flash_attention_decode` is NVPTX-only (its `o[D]` private array crashes SPIR-V), so a real
/// PTX dispatch is the only hardware check this kernel ever gets; `flash_interp.rs`'s sibling test only
/// proves the kernel-IR recipe, not NVPTX codegen/JIT. Position 0 is masked with TRUE `-inf` while the
/// running max is still `-inf` (nothing seen yet): the un-guarded recurrence computed
/// `exp(-inf - -inf) = NaN` here (card 674's fix in `poot-kernelgen/src/flash.rs` routes around it).
#[allow(clippy::too_many_arguments)]
fn flash_decode_ref(
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
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut o = vec![0.0f32; d];
        for t in 0..cap {
            let mut s = 0.0f32;
            for dd in 0..d {
                s += q[h * d + dd] * k[kv * cap * d + t * d + dd];
            }
            s = s * scale + mask[t];
            let m_new = m.max(s);
            let (corr, e) = if m_new == f32::NEG_INFINITY {
                (0.0, 0.0)
            } else {
                ((m - m_new).exp(), (s - m_new).exp())
            };
            l = l * corr + e;
            for dd in 0..d {
                o[dd] = o[dd] * corr + e * v[kv * cap * d + t * d + dd];
            }
            m = m_new;
        }
        for dd in 0..d {
            out[h * d + dd] = o[dd] / l;
        }
    }
    out
}

#[test]
fn flash_decode_leading_true_inf_mask_dispatches_and_matches_cpu_on_ptx() {
    let ctx = match PtxContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!(
                "SKIP flash_decode_leading_true_inf_mask_dispatches_and_matches_cpu_on_ptx: no PTX device ({e})"
            );
            return;
        }
    };

    let (hq, n_rep, cap, d) = (2usize, 1usize, 2usize, 4usize);
    let hkv = hq / n_rep;
    let scale = 0.5f32;
    let fill = |n: usize, seed: u64| -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
            })
            .collect()
    };
    let q = fill(hq * d, 21);
    let k = fill(hkv * cap * d, 22);
    let v = fill(hkv * cap * d, 23);
    let mask = vec![f32::NEG_INFINITY, 0.0];

    let body = kg::flash_attention_decode("flash_decode", hq, n_rep, cap, d, scale, false);
    let dir = std::env::temp_dir().join("poot-ptx-flash-decode-dispatch-probe");
    std::fs::create_dir_all(&dir).unwrap();
    let out_path = artifact_path(&dir, "flash_decode_ptx", Target::Nvptx);
    compile(&body, Target::Nvptx, &out_path).expect("flash_attention_decode must compile to PTX");
    let ptx_bytes = std::fs::read(&out_path).expect("read compiled PTX");
    let kernel = poot_codegen::kernel_handle(&body, Target::Nvptx, ptx_bytes);

    let q_buf = ctx.upload_f32(&q).expect("upload q");
    let k_buf = ctx.upload_f32(&k).expect("upload k");
    let v_buf = ctx.upload_f32(&v).expect("upload v");
    let mask_buf = ctx.upload_f32(&mask).expect("upload mask");
    let out_buf = ctx.alloc_f32(hq * d).expect("allocate out");

    ctx.dispatch_dev(
        "flash_decode_ptx",
        &kernel,
        body.workgroup_size,
        [hq as u32, 1, 1],
        &[&q_buf, &k_buf, &v_buf, &mask_buf],
        &[
            q_buf.elem_count(),
            k_buf.elem_count(),
            v_buf.elem_count(),
            mask_buf.elem_count(),
        ],
        &out_buf,
        out_buf.elem_count(),
    )
    .expect("real PTX flash_attention_decode dispatch");
    let got = ctx.download_f32(&out_buf).expect("download out");

    assert!(
        got.iter().all(|x| x.is_finite()),
        "flash_attention_decode produced non-finite output on real PTX with a leading true-inf mask: {got:?}"
    );
    let want = flash_decode_ref(&q, &k, &v, &mask, hq, n_rep, cap, d, scale);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-4,
        "flash_attention_decode PTX mismatch: got {got:?} want {want:?} (max_abs={max_abs:.3e})"
    );
    eprintln!(
        "flash_attention_decode leading-true-inf-mask OK on real PTX (max_abs={max_abs:.3e})"
    );
}
