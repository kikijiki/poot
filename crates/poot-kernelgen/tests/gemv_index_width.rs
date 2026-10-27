//! POOT-1405: a body the planner narrowed to 32-bit index math ([`IndexWidth::U32`]) lowers its `usize` as i32 on
//! NVPTX, where the native index is i64, and AMDGPU, which has no narrowed lowering, refuses it by name. The
//! generator states the bound the planner narrows under only for the Gemv bodies whose indices it forms itself.

use std::process::Command;

use poot_codegen::{EmitError, Target, emit_llvm_ir};
use poot_kernel_ir::{IndexBound, IndexWidth};
use poot_kernelgen as kg;
use poot_quant::format::{GroupMap, WeightFormat};
use poot_target::{AmdArch, BufferStorage, DeviceCaps};

fn gemv() -> kg::Schedule {
    kg::Schedule::Gemv {
        width: 256,
        cols: 16,
        unroll: 8,
        activation: kg::GemvActivation::Scalar,
    }
}

fn packed(format: WeightFormat, rows: kg::PackedRows) -> kg::KernelRequest {
    kg::KernelRequest::Contraction(kg::ContractionSpec::Packed(kg::PackedContraction {
        format,
        rows,
        schedule: gemv(),
        epilogue: None,
        grid: [4096, 1, 1],
        meta: vec![256, 256],
    }))
}

fn dense(weight: BufferStorage) -> kg::KernelRequest {
    kg::KernelRequest::Contraction(kg::ContractionSpec::DenseGemv {
        weight,
        schedule: gemv(),
        k: 256,
        n: 256,
        epilogue: None,
    })
}

fn body(request: &kg::KernelRequest) -> poot_kernel_ir::Body {
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    };
    kg::generate(request, &DeviceCaps::ptx_default(), &limits)
        .expect("the Gemv request generates")
        .body
}

/// The bound is the Gemv's own: 32 index units per packed word (its bit offsets), the weight lane's elements per
/// word for a dense weight, and the run, column and window overshoot; no claim for a planar format (GPTQ act-order
/// reads group ids from a buffer) or a row-selected table (it reads expert ids).
#[test]
fn the_gemv_states_its_index_bound_only_where_it_forms_every_index_itself() {
    let slack = 256 * 8 + 8 + 256 + 64;
    assert_eq!(
        kg::index_bound(&packed(WeightFormat::Q4_K, kg::PackedRows::Blocks)),
        Some(IndexBound { scale: 32, slack })
    );
    assert_eq!(
        kg::index_bound(&dense(BufferStorage::bf16_packed())),
        Some(IndexBound { scale: 2, slack })
    );
    assert_eq!(
        kg::index_bound(&dense(BufferStorage::f32())),
        Some(IndexBound { scale: 1, slack })
    );
    assert_eq!(
        kg::index_bound(&packed(WeightFormat::Q4_K, kg::PackedRows::Indexed)),
        None
    );
    assert_eq!(
        kg::index_bound(&packed(
            WeightFormat::Gptq {
                groups: GroupMap::Indexed {
                    groups: std::num::NonZeroUsize::new(2).unwrap(),
                },
            },
            kg::PackedRows::Blocks
        )),
        None
    );
}

fn narrowed(request: &kg::KernelRequest) -> poot_kernel_ir::Body {
    let mut body = body(request);
    body.index_width =
        IndexWidth::U32(kg::index_bound(request).expect("the Gemv states its bound"));
    body
}

/// On NVPTX the narrowed body's `usize` is i32: the ABI's i64 lengths are truncated once, thread ids are not
/// widened and element addresses take an i32 index; the native body keeps i64 throughout. With `llc`, the
/// narrowed PTX carries a fraction of the native body's 64-bit integer instructions.
///
/// Mutation: make `Emitter::narrowed` return `false`; the narrowed IR keeps i64 and this row goes red.
#[test]
fn a_narrowed_body_runs_32_bit_index_math_on_nvptx() {
    let request = packed(WeightFormat::Q4_K, kg::PackedRows::Blocks);
    let native = emit_llvm_ir(&body(&request), Target::Nvptx).expect("the native body emits");
    let narrow = emit_llvm_ir(&narrowed(&request), Target::Nvptx).expect("the narrowed body emits");
    assert!(
        native.contains("zext i32") && !native.contains("trunc i64 %n"),
        "the native body widens its thread ids and keeps its i64 lengths"
    );
    assert!(
        narrow.contains("trunc i64 %n") && !narrow.contains("zext i32"),
        "the narrowed body truncates its lengths once and never widens a thread id\n{narrow}"
    );
    let wide_indices: Vec<&str> = narrow
        .lines()
        .filter(|line| line.contains("getelementptr") && line.contains(", i64 %"))
        .collect();
    assert!(
        wide_indices.is_empty(),
        "no element address of the narrowed body takes an i64 index: {wide_indices:#?}"
    );
    if Command::new("llc").arg("--version").output().is_err() {
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
        return;
    }
    let ptx = |ir: &str, tag: &str| {
        let dir = std::env::temp_dir()
            .join("poot-codegen-test")
            .join(format!("gemv-index-{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("k.ll");
        std::fs::write(&ll, ir).unwrap();
        let out = Command::new("llc")
            .args(Target::Nvptx.llc_args())
            .arg(&ll)
            .arg("-o")
            .arg("-")
            .output()
            .expect("llc runs");
        assert!(out.status.success(), "{tag}: llc failed");
        String::from_utf8(out.stdout).unwrap()
    };
    let wide = |ptx: &str| {
        ptx.lines()
            .filter(|line| {
                let op = line.trim_start();
                !op.starts_with("ld.") && !op.starts_with("st.") && !op.starts_with("cvta")
            })
            .filter(|line| line.contains(".s64") || line.contains(".u64") || line.contains(".b64"))
            .count()
    };
    let (native_wide, narrow_wide) = (wide(&ptx(&native, "native")), wide(&ptx(&narrow, "narrow")));
    assert!(
        narrow_wide * 3 < native_wide,
        "the narrowed PTX keeps {narrow_wide} 64-bit integer instructions against the native {native_wide}"
    );
}

/// AMDGPU has no narrowed lowering (`Capability::narrow_index`): a narrowed body is a named refusal, never
/// emitted with a 64-bit index under a 32-bit claim. SPIR-V already indexes in 32 bits and emits it as-is.
///
/// Mutation: drop the narrowed-body guard in `emit_llvm_ir`; the AMDGPU emission succeeds and this row goes red.
#[test]
fn amdgpu_refuses_a_narrowed_body_by_name() {
    let body = narrowed(&dense(BufferStorage::f32()));
    let refused = emit_llvm_ir(&body, Target::AmdGcn(AmdArch::gfx1151()));
    assert!(
        matches!(&refused, Err(EmitError::Unsupported(reason)) if reason.contains("32-bit index width")),
        "got {:?}",
        refused.map(|_| "emitted IR")
    );
    emit_llvm_ir(&body, Target::SpirvVulkan).expect("SPIR-V indexes in 32 bits already");
}
