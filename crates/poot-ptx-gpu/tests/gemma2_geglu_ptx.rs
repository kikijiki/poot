//! Card 536b (R481-007) SC-003, PTX half: Gemma2's GeGLU MLP (`Gelu(gate) * up`, `ops::geglu`) is
//! exactly the pointwise chain this card's fusion-legality fix newly admits into one region -
//! `is_fusable` refused `Gelu` before, so every GeGLU MLP paid for two dispatches (the `Gelu` and the
//! `Mul`) instead of one. Real device regression: compile the decode graph twice against the real PTX
//! device caps - once with `FusionPolicy::Full`, once with `FusionPolicy::MoeHangGuard` - confirm the
//! `Full` compile actually carries a `Fused` region with the `Gelu` step per layer and the guarded
//! compile carries none (so this row proves the fix, not merely "fusion happens somewhere"), then run
//! both compiled graphs on PTX/NVIDIA through the executor contract's resident path
//! (`tests/common::run_resident_with_fusion`, real carried KV cache, zero seeded) and compare dense-decode
//! logits against each other and against the CPU oracle, within tier 2. PTX twin of
//! `poot-gpu/tests/gemma2.rs`'s `gemma2_geglu_fused_decode_matches_unfused_and_cpu_on_gpu` and
//! `poot-rocm-gpu/src/tests/decode.rs`'s `gemma2_geglu_fused_decode_matches_unfused_and_cpu_oracle_rocm`.
//!
//! Card 535a/535b: no caller composes passes itself any more (`compile` owns the one fixed pass order;
//! card 626 deletes the bare `cse`/`fuse` pattern this test used to follow). `FusionPolicy::MoeHangGuard`
//! (cards 186/192's MoE-router-hang guard) is the clean unfused baseline here: it skips exactly
//! `flash_attention_capped` and `fuse` and keeps matmuls off the tiled GEMM (so the compiled graph
//! can carry no `Fused` region), while `canonicalize`, `cse`, `legalize`, the BF16 folds,
//! `lower_nonlast_reduces` and
//! `widen_mismatched_matmul_dtypes` still run identically to `FusionPolicy::Full` - the two compiles diverge only
//! in the fusion-adjacent passes, not in dtype/legality handling, so the comparison below isolates
//! fusion's effect the same way the wgpu/ROCm siblings' `cse`-only-vs-`cse`+`fuse` pair did.
//!
//! Skips cleanly with no NVIDIA GPU; `POOT_REQUIRE_PTX=1` (this crate's require/skip convention, see
//! `poot_runtime_common::DeviceBackend::open_or_skip`) turns that skip into a failure instead.
//!
//! Mutation (recorded in the card's landing note, never left in the tree): drop `Exp` from
//! `poot_graph_plan::FUSABLE_FLOAT_UNARY_OPS` (`fuse.rs`) - the GeGLU region assertion on the
//! `FusionPolicy::Full` compile below fails immediately (the GELU chain splits at its `exp`, so no
//! `Fused` region carries the whole GELU any more), before the device even runs.

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::Device;
use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_graph_ir::op::{FusedOp, FusedOperand, OpKind, UnOp};
use poot_graph_ir::{Slot, Storage, ValueId};
use poot_graph_plan::{CompileOptions, FusionPolicy, Submission, Target, compile};
use poot_models::model::{LogitRows, Phase};
use poot_ptx_gpu::PtxDevice;
use poot_runtime_common::DeviceBackend;
use poot_tensor::HostTensor;
use poot_test_util::{assert_close_rel, max_abs_error, seed_of};

mod common;

/// Deterministic pseudo-random fill in `[-0.1, 0.1)`. Local copy of the wgpu/ROCm twins' `fill`
/// (integration-test crates cannot reach `poot-eval`'s `pub(super)` helper).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 0.1
        })
        .collect()
}

/// Does this fused region carry the tanh-approximate GELU (`ops::gelu`) of a GeGLU MLP: its `exp` step and
/// its `0.044715` cubic coefficient. Both are required, so splitting the chain at `exp` (a region that
/// keeps only the polynomial half) is not counted.
fn carries_gelu(region: &poot_graph_ir::op::FusedRegion) -> bool {
    let has_exp = region
        .steps
        .iter()
        .any(|s| matches!(s.op, FusedOp::Unary(UnOp::Exp)));
    let has_coeff = region.steps.iter().any(|s| {
        s.inputs
            .contains(&FusedOperand::Lit(poot_graph_ir::Scalar::F32(0.044715)))
    });
    has_exp && has_coeff
}

/// Count `Fused` regions in `g` that carry the GeGLU GELU.
fn geglu_region_count(g: &poot_graph_ir::Graph) -> usize {
    g.eqns
        .iter()
        .filter(|e| match &e.op {
            OpKind::Fused(region) => carries_gelu(region),
            _ => false,
        })
        .count()
}

#[test]
fn gemma2_geglu_fused_decode_matches_unfused_and_cpu_oracle_ptx() {
    let Some(mut ptx) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };

    // A tiny Gemma 2 (4 layers, GeGLU, both logit softcaps, a 4-key window) traced for one decode
    // token over 8 cached positions by the registry's family.
    let layers = 4usize;
    let pos = 3usize;
    let cap = 8usize;
    let m = Dense::new(Family::Gemma2)
        .vocab(64)
        .dims(64, 128, layers)
        .heads(4, 2)
        .head_dim(16)
        .max_positions(32)
        .with("sliding_window", 4)
        .with("attn_logit_softcapping", 50.0)
        .with("final_logit_softcapping", 30.0)
        .with("query_pre_attn_scalar", serde_json::Value::Null)
        .f32_model();
    let g = plain(
        m.model
            .trace(Phase::Decode, step(1, 1, cap, LogitRows::Last))
            .unwrap(),
    );

    // Bind the slots and Const inputs against the raw traced graph; `compile` preserves `g`'s
    // input/const identity (card 535b), so this bind stays valid for both compiled graphs below.
    // `Storage::State` (the KV cache) is zero at first declaration (Z5) and bound by the contract.
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for (i, &id) in g.inputs.iter().enumerate() {
        let m = g.meta(id);
        match m.storage {
            Storage::State | Storage::Computed(_) => continue,
            Storage::Slot(slot) => {
                let v = match slot {
                    Slot::Token => 5,
                    Slot::Pos => pos as i32,
                    other => panic!("unexpected slot {other:?} on gemma2 decode"),
                };
                inputs.insert(id, HostTensor::i32(m.aval.shape.clone(), vec![v]).into());
            }
            _ => {
                let seed = seed_of(m.name.as_deref().unwrap_or("")).wrapping_add(100 + i as u64);
                inputs.insert(
                    id,
                    HostTensor::f32(m.aval.shape.clone(), fill(m.aval.numel(), seed)).into(),
                );
            }
        }
    }

    // Card 522: compile against the real, measured PTX device, never a default target.
    let target = Target {
        backend: poot_target::Backend::Nvptx,
        caps: Device::target(&ptx).caps,
    };
    let fused = compile(
        &g,
        &target,
        &CompileOptions {
            execution: Submission::Replay,
            fusion: FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .expect("compile the gemma2 GeGLU decode graph, FusionPolicy::Full");
    assert!(
        fused.passes().any(|p| p == "fuse"),
        "FusionPolicy::Full must run fuse"
    );
    let fused_geglu_count = geglu_region_count(fused.graph());
    assert_eq!(
        fused_geglu_count,
        layers,
        "expected one Fused region carrying the GeGLU Gelu step per layer: {} eqns {:?}",
        fused.graph().eqns.len(),
        fused
            .graph()
            .eqns
            .iter()
            .map(|e| e.op.name())
            .collect::<Vec<_>>()
    );

    let unfused = compile(
        &g,
        &target,
        &CompileOptions {
            execution: Submission::Replay,
            fusion: FusionPolicy::MoeHangGuard,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .expect("compile the gemma2 GeGLU decode graph, FusionPolicy::MoeHangGuard (unfused baseline)");
    assert!(
        !unfused.passes().any(|p| p == "fuse"),
        "FusionPolicy::MoeHangGuard must skip fuse"
    );
    assert_eq!(
        geglu_region_count(unfused.graph()),
        0,
        "the MoeHangGuard baseline must carry no Fused region"
    );
    assert!(
        fused.graph().eqns.len() < unfused.graph().eqns.len(),
        "fewer dispatches after fusion: {} -> {}",
        unfused.graph().eqns.len(),
        fused.graph().eqns.len()
    );

    // `compile`'s own `Program`s are only needed for the structural assertions above; the device run
    // goes through the executor contract's own staging (Card 549), from the raw traced graph, with the
    // matching fusion policy for each side.
    drop(unfused);
    drop(fused);

    // Real carried KV cache, zero-seeded (this is the graph's first decode step in isolation - `pos`
    // names where in the RoPE table/valid-prefix slice the step reads, not how many prior steps ran).
    // The executor contract's own state buffers are always zero at first declaration (Z5), so this
    // matches `common::run_resident_with_fusion`'s state exactly with no seeding of its own.
    let zero_state: Vec<HostTensor> = g
        .state
        .iter()
        .map(|&(si, _)| HostTensor::zeros(g.aval(si).shape.clone()))
        .collect();

    let mut cpu_inputs = inputs.clone();
    for &id in &g.inputs {
        if let Storage::Computed(computed) = g.meta(id).storage {
            cpu_inputs.insert(
                id,
                HostTensor::f32(computed.shape(), computed.values_f32()).into(),
            );
        }
    }
    for (i, &(si, _)) in g.state.iter().enumerate() {
        cpu_inputs.insert(si, zero_state[i].clone().into());
    }
    let cpu_eval = eval(&g, &cpu_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("cpu oracle eval_with_state");
    let cpu = cpu_eval.output.into_host().expect("dense output");
    let _cpu_new_state: Vec<HostTensor> = cpu_eval
        .state
        .into_iter()
        .map(Value::into_host)
        .collect::<Result<Vec<_>, _>>()
        .expect("dense state");
    let unfused_ptx =
        common::run_resident_with_fusion(&mut ptx, &g, &inputs, FusionPolicy::MoeHangGuard);
    let fused_ptx = common::run_resident_with_fusion(&mut ptx, &g, &inputs, FusionPolicy::Full);

    assert_eq!(fused_ptx.shape(), unfused_ptx.shape());
    assert_eq!(fused_ptx.shape(), cpu.shape());
    // Fusion moves region intermediates from f32 global buffers to registers, so fused vs un-fused is a
    // tight fp rounding difference (tier 1 territory), while GPU vs CPU is the ordinary tier-2 gap.
    let fused_vs_unfused =
        max_abs_error(fused_ptx.as_f32().unwrap(), unfused_ptx.as_f32().unwrap());
    assert!(
        fused_vs_unfused <= 1e-4,
        "fused PTX decode must match un-fused within fp tolerance: max_abs={fused_vs_unfused:.3e}"
    );
    assert_close_rel(fused_ptx.as_f32().unwrap(), cpu.as_f32().unwrap(), 5e-3);
    eprintln!(
        "gemma2_geglu_fused_decode_matches_unfused_and_cpu_oracle_ptx: fused/unfused max_abs={:.3e}, \
         fused/cpu max_abs={:.3e}",
        fused_vs_unfused,
        max_abs_error(fused_ptx.as_f32().unwrap(), cpu.as_f32().unwrap())
    );
}
