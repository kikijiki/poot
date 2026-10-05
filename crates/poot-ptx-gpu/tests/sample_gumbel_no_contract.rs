//! Card 675 SC-001 (ADR 0114 tier 1), PTX lane: on real NVPTX hardware, the production Gumbel-max
//! sampler kernel's perturbed-argmax pick must be bit-exact against the CPU oracle on a fixture
//! engineered so that fusing the feeding multiply into the add (one rounding instead of two) would flip
//! the winning index. Loads the real, committed `poot-graph-plan` asset (the body the engine actually
//! ships, already carrying `no_contract_add`'s `Rvalue::BinaryOpNoContract` - see
//! `crates/pootc/tests/import_run.rs`'s `gumbel_select_nvptx_add_is_marked_no_contract_and_isa_guaranteed_unfused`
//! for the IR-level mutation that proves the marker is wired), not a re-import through `pootc` (this
//! crate has no rustc-linked toolchain and the pod has no cargo to rebuild one).
//!
//! Fixture: lane 0's `noise_scale=1.9793916, noise=29.984547, logit(scaled)=79.44914` two-step-rounds to
//! `138.800293...`, exactly one ULP below `noise_scale.mul_add(noise, logit)` = `138.800308...`; lane 1's
//! own `noise_scale * 0.0` term is exactly zero, so its perturbed value is just its logit, set to that
//! `mul_add` result bit for bit. Two-step (correct, what this kernel must compute): lane 1 > lane 0, lane
//! 1 wins. One rounding (if the add were ever fused): lane 0 == lane 1 exactly, and the `(value desc,
//! index asc)` tie rule hands it to the lower index - lane 0, flipping the winner.
//!
//! Per card 675 (matching card 628's own SC-001): today's llc/ptxas do not contract poot-codegen's plain float path (explicit `.rn` on every
//! add/mul; see `pootc/tests/import_run.rs`'s companion IR/PTX-text test), so there is no "unmarked
//! diverges" row here - that mutation lives in the marker's own IR contract
//! (`BinaryOpNoContract` -> `llvm.experimental.constrained.fadd.f32` + `strictfp`), not in this dispatch.
//! This test only has to prove the GREEN: the real, marked, production kernel is bit-exact on real
//! hardware.
//!
//! `PtxContext::new()` implements the project's device-skip convention (AGENTS.md): `Err` with no NVIDIA
//! driver, panic instead when `POOT_REQUIRE_PTX=1` (set by `just test-device-ptx` and the
//! PTX pod runs), so a lane that was supposed to run this never reports a skip as a pass.

use poot_codegen::{Target, artifact_path, compile, kernel_handle};
use poot_kernel_ir::Body;
use poot_ptx_runtime::{BufferRole, BufferStorage, PtxContext};

/// The committed production asset `poot-graph-plan` ships for `SampleRule::Gumbel`
/// (`crates/poot-graph-plan/assets/sample_gumbel_argmax_batched.kir.json`), embedded at compile time so
/// the test binary is self-contained (no source tree, no pootc/cargo, needed on a pod).
const GUMBEL_ASSET: &str =
    include_str!("../../poot-graph-plan/assets/sample_gumbel_argmax_batched.kir.json");

#[test]
fn gumbel_select_nvptx_fma_sensitive_fixture_matches_cpu_oracle() {
    let body: Body =
        serde_json::from_str(GUMBEL_ASSET).expect("deserialize the committed Gumbel asset");
    assert_eq!(
        body.param_count, 5,
        "sample_gumbel_argmax_batched has 5 slice params (logits, noise, params, dims, out)"
    );
    let no_contract_count = body
        .blocks
        .iter()
        .flat_map(|bb| &bb.statements)
        .filter(|s| {
            matches!(
                s,
                poot_kernel_ir::Statement::Assign(
                    _,
                    poot_kernel_ir::Rvalue::BinaryOpNoContract(poot_kernel_ir::BinOp::Add, _, _)
                )
            )
        })
        .count();
    assert_eq!(
        no_contract_count, 1,
        "the committed Gumbel asset should carry exactly one no-contract Add (the perturbed-argmax sum); \
         got {no_contract_count} - card 675's marker may not have been regenerated into this asset"
    );

    let dir = std::env::temp_dir().join("poot-ptx-gpu-gumbel-no-contract");
    std::fs::create_dir_all(&dir).unwrap();
    let out = artifact_path(&dir, "sample_gumbel_argmax_batched", Target::Nvptx);
    compile(&body, Target::Nvptx, &out).expect("Gumbel asset must compile to NVPTX");
    let ptx_bytes = std::fs::read(&out).expect("read compiled PTX");
    let kernel = kernel_handle(&body, Target::Nvptx, ptx_bytes);

    let ctx = match PtxContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "SKIP gumbel_select_nvptx_fma_sensitive_fixture_matches_cpu_oracle: no PTX device ({e})"
            );
            return;
        }
    };

    // vocab=2: index 0 is the FMA-sensitive term; index 1 is the fixed competitor whose own two-step
    // value lands exactly on `noise_scale.mul_add(noise0, logit0)` via noise1=0.0.
    let noise_scale = 1.979_391_6_f32;
    let noise0 = 29.984_547_f32;
    let logit0 = 79.449_14_f32; // inv_temp = 1.0, so scaled0 = logit0 exactly
    let logit1 = noise_scale.mul_add(noise0, logit0);
    let two_step0 = logit0 + noise_scale * noise0;
    assert_ne!(
        two_step0.to_bits(),
        logit1.to_bits(),
        "fixture requires a real ULP gap between the two-step and fused values"
    );
    assert!(
        two_step0 < logit1,
        "fixture assumes the two-step value rounds below the fused value here"
    );

    let logits = [logit0, logit1];
    let noise = [noise0, 0.0f32];
    let params = [1.0f32, -1000.0, noise_scale]; // inv_temp, floor_offset (both stay above floor), noise_scale
    let want = (1i32, -1i32); // the CPU two-step oracle's pick: lane 1 (strictly higher), no non-finite logit

    let logits_buf = ctx.upload_f32(&logits).expect("upload logits");
    let noise_buf = ctx.upload_f32(&noise).expect("upload noise");
    let params_buf = ctx.upload_f32(&params).expect("upload params");
    let dims_buf = ctx
        .upload_i32(&[logits.len() as i32]) // dims is u32 on-device; bit-identical as i32 here
        .expect("upload dims");
    let out_buf = ctx
        .alloc_storage(BufferRole::Activation, BufferStorage::i32(), 2)
        .expect("alloc out");
    ctx.dispatch_dev(
        "sample_gumbel_argmax_batched",
        &kernel,
        [64, 1, 1],
        [64, 1, 1],
        &[&logits_buf, &noise_buf, &params_buf, &dims_buf],
        &[
            logits_buf.elem_count(),
            noise_buf.elem_count(),
            params_buf.elem_count(),
            dims_buf.elem_count(),
        ],
        &out_buf,
        out_buf.elem_count(),
    )
    .expect("dispatch sample_gumbel_argmax_batched on PTX");
    let got = ctx.download_i32(&out_buf).expect("download out");
    let got = (got[0], got[1]);

    assert_eq!(
        got, want,
        "card 675 SC-001: the production (marked) Gumbel kernel must match the CPU two-step oracle on \
         this FMA-sensitive fixture on real NVPTX hardware"
    );
    eprintln!(
        "card 675 SC-001 GREEN: picked {got:?} on real NVPTX hardware, matching the CPU oracle's \
         two-step computation"
    );
}
