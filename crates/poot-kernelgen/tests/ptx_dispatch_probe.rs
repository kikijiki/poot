//! Card 530 (owner PTX is first class, not compile-only-verified): real NVIDIA dispatch of
//! `wmma_tile` (the target-neutral single 16x16x16 fragment tile, F16 operands / F32 accumulate), compiled
//! to real PTX (`poot_codegen::compile`, `Target::Nvptx`) and run via `poot_ptx_runtime::PtxContext`'s
//! `dispatch_dev`. The NVPTX sibling of `poot-kernelgen`'s `coopmat_dispatch_probe` (wgpu/RADV coopmat) and
//! `poot-rocm-device`'s `tests::resident::probe_*_wmma_rocm` (ROCm WMMA): all three exercise the same fragment
//! op lowered to a different target and compare against the same CPU oracle.
//!
//! `PtxContext::new()` already implements the project's device-skip convention (AGENTS.md): it returns
//! `Err` when no NVIDIA driver/device is present, and panics instead when `POOT_REQUIRE_PTX=1` (set by
//! `just test-device-ptx` and the PTX pod runs), so this test never reports a skip as a
//! pass on a lane that was supposed to run it.

use poot_codegen::{Target, artifact_path, compile};
use poot_kernelgen as kg;
use poot_ptx_runtime::PtxContext;

#[path = "support/attention.rs"]
mod attention;
use attention::{KeyExtras, attention_reference, attention_reference_keys};

/// Encode f32 -> f16 bytes (little-endian), reusing the loader's own f32->f16 narrowing
/// (`poot_quant::scalar::f32_to_f16`), matching `coopmat_dispatch_probe.rs`'s helper.
fn f16_encode(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|&f| poot_quant::scalar::f32_to_f16(f).to_le_bytes())
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

    let body = poot_kernel_ir::test_support::wmma_tile("k");
    let dir = std::env::temp_dir().join("poot-ptx-wmma-dispatch-probe");
    std::fs::create_dir_all(&dir).unwrap();
    let out = artifact_path(&dir, "wmma_tile_ptx", Target::Nvptx);
    compile(&body, Target::Nvptx, &out).expect("wmma_tile must compile to PTX");
    let ptx_bytes = std::fs::read(&out).expect("read compiled PTX");
    let kernel = poot_codegen::kernel_handle(&body, poot_codegen::GpuTarget::Nvptx, ptx_bytes);

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
    let round = |x: f32| poot_quant::scalar::f16_to_f32(poot_quant::scalar::f32_to_f16(x));
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

/// Generate `spec` for a PTX device, compile it to PTX and run it on `ctx` over its own launch.
fn run_attention_on_ptx(
    ctx: &PtxContext,
    name: &str,
    spec: &kg::AttentionSpec,
    operands: [&[f32]; 4],
) -> Vec<f32> {
    run_attention_over_keys_on_ptx(ctx, name, spec, operands, None, None)
}

/// [`run_attention_on_ptx`] with a split key's shared part and a live bound's per-row lengths, the kernel's
/// parameters after `mask` in that order.
fn run_attention_over_keys_on_ptx(
    ctx: &PtxContext,
    name: &str,
    spec: &kg::AttentionSpec,
    [q, k, v, mask]: [&[f32]; 4],
    k_shared: Option<&[f32]>,
    len: Option<&[i32]>,
) -> Vec<f32> {
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
    };
    let generated = kg::generate(
        &kg::KernelRequest::Attention(spec.clone()),
        &poot_target::DeviceCaps::ptx_default(),
        &limits,
    )
    .expect("the attention request generates");
    let body = generated.body;
    let dir = std::env::temp_dir().join("poot-ptx-attention-dispatch-probe");
    std::fs::create_dir_all(&dir).unwrap();
    let out_path = artifact_path(&dir, name, Target::Nvptx);
    compile(&body, Target::Nvptx, &out_path).expect("the attention kernel must compile to PTX");
    let ptx_bytes = std::fs::read(&out_path).expect("read compiled PTX");
    let kernel = poot_codegen::kernel_handle(&body, poot_codegen::GpuTarget::Nvptx, ptx_bytes);

    let q_buf = ctx.upload_f32(q).expect("upload q");
    let k_buf = ctx.upload_f32(k).expect("upload k");
    let v_buf = ctx.upload_f32(v).expect("upload v");
    let mask_buf = ctx.upload_f32(mask).expect("upload mask");
    let shared_buf = k_shared.map(|data| ctx.upload_f32(data).expect("upload k_shared"));
    let len_buf = len.map(|data| ctx.upload_i32(data).expect("upload len"));
    let out_len = spec.batch * spec.hq * spec.lq * spec.dv;
    let out_buf = ctx.alloc_f32(out_len).expect("allocate out");
    let mut buffers = vec![&q_buf, &k_buf, &v_buf, &mask_buf];
    buffers.extend(shared_buf.as_ref());
    buffers.extend(len_buf.as_ref());
    let counts: Vec<u32> = buffers.iter().map(|buffer| buffer.elem_count()).collect();
    ctx.dispatch_dev(
        name,
        &kernel,
        body.workgroup_size,
        generated.launch.grid,
        &buffers,
        &counts,
        &out_buf,
        out_buf.elem_count(),
    )
    .unwrap_or_else(|e| panic!("{name}: real PTX attention dispatch: {e:?}"));
    ctx.download_f32(&out_buf).expect("download out")
}

fn attention_fill(n: usize, seed: u64) -> Vec<f32> {
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

/// The streaming attention spec for `[batch, hq, lq, d]` queries against `lk` keys, `hq / n_rep` kv heads.
#[allow(clippy::too_many_arguments)]
fn attention_spec(
    (batch, hq, n_rep, lq, lk, d): (usize, usize, usize, usize, usize, usize),
    lanes: usize,
    mask: kg::MaskKind,
    score: kg::ScoreTransform,
) -> kg::AttentionSpec {
    kg::AttentionSpec {
        batch,
        hq,
        n_rep,
        lq,
        lk,
        dk: d,
        dv: d,
        key: kg::KeyLayout::Single,
        score,
        mask,
        bound: kg::KeyBound::Static,
        schedule: kg::AttentionSchedule::Streaming { lanes },
    }
}

/// A real PTX dispatch of the generated attention kernel, the only hardware check of its NVPTX codegen: the
/// kernel-IR interpreter rows prove the recipe, not the lowering or the JIT. Position 0 is masked with TRUE
/// `-inf` while every lane's running maximum is still `-inf`, and 4 lanes share 2 keys so two lanes see no
/// key at all: the unguarded recurrence computed `exp(-inf - -inf) = NaN` for both.
#[test]
fn attention_leading_true_inf_mask_dispatches_and_matches_cpu_on_ptx() {
    let ctx = match PtxContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!(
                "SKIP attention_leading_true_inf_mask_dispatches_and_matches_cpu_on_ptx: no PTX device ({e})"
            );
            return;
        }
    };
    let (hq, n_rep, cap, d) = (2usize, 1usize, 2usize, 4usize);
    let spec = attention_spec(
        (1, hq, n_rep, 1, cap, d),
        4,
        kg::MaskKind::Broadcast,
        kg::ScoreTransform::Scale {
            scale_bits: 0.5f32.to_bits(),
        },
    );
    let q = attention_fill(hq * d, 21);
    let k = attention_fill(hq / n_rep * cap * d, 22);
    let v = attention_fill(hq / n_rep * cap * d, 23);
    let mask = vec![f32::NEG_INFINITY, 0.0];
    let got = run_attention_on_ptx(&ctx, "attention_ptx", &spec, [&q, &k, &v, &mask]);
    assert!(
        got.iter().all(|x| x.is_finite()),
        "the attention kernel produced non-finite output on real PTX with a leading true-inf mask: {got:?}"
    );
    let want = attention_reference(&spec, &q, &k, &v, &mask);
    let max_abs = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-4,
        "attention PTX mismatch: got {got:?} want {want:?} (max_abs={max_abs:.3e})"
    );
    eprintln!("attention leading-true-inf-mask OK on real PTX (max_abs={max_abs:.3e})");
}

/// The fixture bank on real PTX, each row with more keys than lanes so lanes walk several keys and the
/// running state is rescaled between them: Llama GQA decode at `B = 1` and `B = 4` (D = 64, 64 lanes over 200
/// keys), D = 512 decode (16 lanes over 96 keys, the head width SPIR-V used to refuse), a softcapped Gemma-style
/// prefill (D = 128, 64 lanes over 100 rows), and an ALiBi per-head-mask decode. Each against the textbook
/// two-pass attention.
#[test]
fn attention_fixture_bank_matches_cpu_on_ptx() {
    let ctx = match PtxContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("SKIP attention_fixture_bank_matches_cpu_on_ptx: no PTX device ({e})");
            return;
        }
    };
    let scale = |d: usize| kg::ScoreTransform::Scale {
        scale_bits: (1.0 / (d as f32).sqrt()).to_bits(),
    };
    let rows = [
        (
            "llama_decode_b1",
            attention_spec(
                (1, 8, 4, 1, 200, 64),
                64,
                kg::MaskKind::Broadcast,
                scale(64),
            ),
        ),
        (
            "llama_decode_b4",
            attention_spec(
                (4, 8, 4, 1, 200, 64),
                64,
                kg::MaskKind::Broadcast,
                scale(64),
            ),
        ),
        (
            "decode_d512",
            attention_spec(
                (1, 2, 1, 1, 96, 512),
                16,
                kg::MaskKind::Broadcast,
                scale(512),
            ),
        ),
        (
            "gemma_softcap_prefill",
            attention_spec(
                (1, 4, 2, 100, 100, 128),
                64,
                kg::MaskKind::Broadcast,
                kg::ScoreTransform::ScaleSoftcap {
                    scale_bits: (1.0 / (128f32).sqrt()).to_bits(),
                    cap_bits: 50.0f32.to_bits(),
                },
            ),
        ),
        (
            "prefill_into_a_cache",
            attention_spec(
                (1, 4, 2, 3, 100, 128),
                64,
                kg::MaskKind::Broadcast,
                scale(128),
            ),
        ),
        (
            "alibi_decode",
            attention_spec((2, 8, 2, 1, 150, 64), 64, kg::MaskKind::PerHead, scale(64)),
        ),
    ];
    for (name, spec) in rows {
        let hkv = spec.hq / spec.n_rep;
        let q = attention_fill(spec.batch * spec.hq * spec.lq * spec.dk, 1);
        let k = attention_fill(spec.batch * hkv * spec.lk * spec.dk, 2);
        let v = attention_fill(spec.batch * hkv * spec.lk * spec.dv, 3);
        let mask_heads = match spec.mask {
            kg::MaskKind::Broadcast => 1,
            kg::MaskKind::PerHead => spec.hq,
            kg::MaskKind::Blocks { .. } => unreachable!("no block-selected row in this table"),
        };
        let mask: Vec<f32> = (0..spec.batch * mask_heads * spec.lq * spec.lk)
            .map(|i| {
                let (key, row) = (i % spec.lk, (i / spec.lk) % spec.lq);
                let (plane, head) = (
                    i / (spec.lq * spec.lk),
                    (i / (spec.lq * spec.lk)) % mask_heads,
                );
                // Decode rows hide a different tail per plane; prefill rows are causal; per-head planes carry a
                // per-head ALiBi-style bias.
                let visible = if spec.lq == 1 {
                    key < spec.lk - 3 * (plane / mask_heads)
                } else {
                    // Prefill is causal over the cache tail: row `i` sees the prefix and rows up to itself.
                    key <= spec.lk - spec.lq + row
                };
                if !visible {
                    -1.0e9
                } else if spec.mask == kg::MaskKind::PerHead {
                    -0.05 * (head as f32 + 1.0) * (spec.lk - key) as f32 / spec.lk as f32
                } else {
                    0.0
                }
            })
            .collect();
        let got = run_attention_on_ptx(&ctx, name, &spec, [&q, &k, &v, &mask]);
        let want = attention_reference(&spec, &q, &k, &v, &mask);
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(g, w)| {
                if (g - w).is_nan() {
                    f32::INFINITY
                } else {
                    (g - w).abs()
                }
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "{name}: attention PTX mismatch (max_abs={max_abs:.3e})"
        );
        eprintln!("attention {name} OK on real PTX (max_abs={max_abs:.3e})");
    }
}

/// The split key and the live bound on real PTX, the only hardware check of their NVPTX codegen: a DeepSeek-V2
/// MLA row (`dk` 192 = 128 per head + 64 shared, `dv` 128, 8 lanes over 100 keys) with a per-row live length,
/// NaN in every key and value past it, and lengths outside `1..=Lk` (a NaN row for a length below one, every key
/// for a length past the capacity, as the decomposition selects). Against the textbook attention over each
/// row's own keys.
#[test]
fn attention_split_key_and_live_bound_match_cpu_on_ptx() {
    let ctx = match PtxContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!(
                "SKIP attention_split_key_and_live_bound_match_cpu_on_ptx: no PTX device ({e})"
            );
            return;
        }
    };
    let (batch, hq, lk, dk, dv, shared) = (4usize, 2usize, 100usize, 192usize, 128usize, 64usize);
    let head = dk - shared;
    let spec = kg::AttentionSpec {
        batch,
        hq,
        n_rep: 1,
        lq: 1,
        lk,
        dk,
        dv,
        key: kg::KeyLayout::Split { shared_dim: shared },
        score: kg::ScoreTransform::ScaleSoftcap {
            scale_bits: 0.07f32.to_bits(),
            cap_bits: 30.0f32.to_bits(),
        },
        mask: kg::MaskKind::Broadcast,
        bound: kg::KeyBound::Live,
        schedule: kg::AttentionSchedule::Streaming { lanes: 8 },
    };
    let len = [37i32, 100, -1, 1000];
    let q = attention_fill(batch * hq * dk, 1);
    let mut k = attention_fill(batch * hq * lk * head, 2);
    let mut v = attention_fill(batch * hq * lk * dv, 3);
    let mut k_shared = attention_fill(batch * lk * shared, 4);
    for (b, &live) in len.iter().enumerate() {
        for t in (live.max(0) as usize).min(lk)..lk {
            // Rows past a valid length are poisoned; a row whose length is outside the range is read by the
            // kernel per the clamp, never past the buffers.
            if (1..=lk as i32).contains(&live) {
                for h in 0..hq {
                    let row = (b * hq + h) * lk + t;
                    k[row * head..(row + 1) * head].fill(f32::NAN);
                    v[row * dv..(row + 1) * dv].fill(f32::NAN);
                }
                k_shared[(b * lk + t) * shared..(b * lk + t + 1) * shared].fill(f32::NAN);
            }
        }
    }
    let mask = vec![0.0f32; batch * lk];
    let got = run_attention_over_keys_on_ptx(
        &ctx,
        "attention_split_live_ptx",
        &spec,
        [&q, &k, &v, &mask],
        Some(&k_shared),
        Some(&len),
    );
    let want = attention_reference_keys(
        &spec,
        &q,
        &k,
        &v,
        &mask,
        KeyExtras {
            k_shared: Some(&k_shared),
            len: Some(&len),
            ..KeyExtras::default()
        },
    );
    let row = hq * dv;
    for (b, &live) in len.iter().enumerate() {
        let (got, want) = (&got[b * row..(b + 1) * row], &want[b * row..(b + 1) * row]);
        if live < 1 {
            assert!(got.iter().all(|x| x.is_nan()), "len {live}: {got:?}");
            continue;
        }
        let max_abs = got
            .iter()
            .zip(want)
            .map(|(g, w)| {
                if (g - w).is_nan() {
                    f32::INFINITY
                } else {
                    (g - w).abs()
                }
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "len {live}: PTX mismatch (max_abs={max_abs:.3e})"
        );
        eprintln!("attention split+live len {live} OK on real PTX (max_abs={max_abs:.3e})");
    }
}
