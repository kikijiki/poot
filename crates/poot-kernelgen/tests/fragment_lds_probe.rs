//! POOT-1432: fragment-IR completeness on every backend. The `FragmentProbe` body stages `A`/`B` tiles in LDS,
//! loads them at a nonzero base in both layouts (`WmmaLoadLds`), runs a K loop of multiplies, optionally seeds
//! the accumulator with an accumulator `WmmaLoadLds`, and reads the accumulator back by `WmmaStore`,
//! `WmmaStoreLds` or the declared element map (`WmmaElemRead`/`WmmaElemWrite`, AMDGCN). Every dispatch is
//! checked against one host reference. The operands are multiples of 1/16 in [-1/2, 1/2], so the narrow
//! conversion is the identity and every product is exact; the device accumulates in its own order and rounding
//! (a few f32 ulp of the partial sums, below 1e-5 here), while a wrong base, layout, stride or lane map moves an
//! element by at least 1/256. The tolerance is `TOLERANCE` = 1e-4, so a layout error is a failure, never noise.
//!
//! The compile rows need only the toolchain; the `_on_*` rows dispatch on the device and skip (loudly) without
//! one. PTX rows run on a pod.

use poot_codegen::{Target, artifact_path, compile, compile_kernel, kernel_handle};
use poot_kernel_ir::test_support::{AccInit, FragmentProbe, Readout};
use poot_kernel_ir::{FragmentMajor, WmmaDtype};
use poot_runtime_common::BufferRole;
use poot_target::BufferStorage;

use FragmentMajor::{Col, Row};

fn have(tool: &str) -> bool {
    std::process::Command::new(tool)
        .arg("--version")
        .output()
        .is_ok()
}

/// Largest accepted |device - reference|: well above the device's accumulation rounding, well below the 1/256
/// step by which any misplaced element differs.
const TOLERANCE: f32 = 1e-4;

/// Operand and accumulator-seed values: multiples of 1/16 in [-1/2, 1/2] (exact in f16 and bf16).
fn value(salt: usize, i: usize) -> f32 {
    ((i * 7 + salt * 5) % 17) as f32 / 16.0 - 0.5
}

/// One probe's inputs in buffer order and the expected output in output order.
struct Case {
    a: Vec<f32>,
    b: Vec<f32>,
    c0: Vec<f32>,
    expect: Vec<f32>,
}

impl Case {
    fn new(spec: &FragmentProbe) -> Case {
        let tiles = spec.k_tiles as usize;
        let e = FragmentProbe::EDGE;
        let logical_a = |t: usize, m: usize, k: usize| value(1, t * 256 + m * 16 + k);
        let logical_b = |t: usize, k: usize, n: usize| value(2, t * 256 + k * 16 + n);
        let logical_c = |m: usize, n: usize| match spec.acc_init {
            AccInit::Zero => 0.0,
            AccInit::LoadLds(_) => value(3, m * 16 + n),
        };
        let mut a = vec![f32::NAN; tiles * 256];
        let mut b = vec![f32::NAN; tiles * 256];
        let mut c0 = vec![0.0; 256];
        let mut expect = vec![f32::NAN; 256];
        for t in 0..tiles {
            for r in 0..e {
                for c in 0..e {
                    a[spec.a_index(t, r, c)] = logical_a(t, r, c);
                    b[spec.b_index(t, r, c)] = logical_b(t, r, c);
                }
            }
        }
        for m in 0..e {
            for n in 0..e {
                c0[spec.c0_index(m, n)] = logical_c(m, n);
                let mut acc = logical_c(m, n);
                for t in 0..tiles {
                    for k in 0..e {
                        acc += logical_a(t, m, k) * logical_b(t, k, n);
                    }
                }
                expect[spec.out_index(m, n)] = acc * spec.row_scale(m);
            }
        }
        Case { a, b, c0, expect }
    }

    fn operand_bytes(spec: &FragmentProbe, values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| match spec.dtype {
                WmmaDtype::F16 => poot_quant::scalar::f32_to_f16(v).to_le_bytes(),
                WmmaDtype::Bf16 => poot_quant::scalar::f32_to_bf16(v).to_le_bytes(),
            })
            .collect()
    }

    fn check(&self, label: &str, got: &[f32]) {
        assert_eq!(got.len(), self.expect.len(), "{label}: output length");
        for (i, (g, w)) in got.iter().zip(&self.expect).enumerate() {
            assert!(
                (g - w).abs() <= TOLERANCE,
                "{label}: out[{i}] (row {}, col {}) = {g}, expected {w}",
                i / 16,
                i % 16
            );
        }
    }
}

fn probe(
    dtype: WmmaDtype,
    (a_major, b_major): (FragmentMajor, FragmentMajor),
    k_tiles: u32,
    acc_init: AccInit,
    readout: Readout,
) -> FragmentProbe {
    FragmentProbe {
        dtype,
        a_major,
        b_major,
        k_tiles,
        acc_init,
        readout,
    }
}

/// The configurations every backend runs: both layouts of `A` and `B`, a three-step K loop, an accumulator
/// seeded from LDS and read back through LDS in both layouts.
fn portable_probes(dtype: WmmaDtype) -> Vec<FragmentProbe> {
    let mut probes = Vec::new();
    for majors in [(Row, Row), (Row, Col), (Col, Row), (Col, Col)] {
        probes.push(probe(dtype, majors, 3, AccInit::Zero, Readout::Store));
    }
    probes.push(probe(
        dtype,
        (Row, Col),
        2,
        AccInit::LoadLds(Col),
        Readout::StoreLds(Row),
    ));
    probes.push(probe(
        dtype,
        (Col, Row),
        2,
        AccInit::LoadLds(Row),
        Readout::StoreLds(Col),
    ));
    probes
}

/// The AMDGCN-only configurations: the accumulator read and rewritten through the declared element map.
fn declared_probes(dtype: WmmaDtype) -> Vec<FragmentProbe> {
    vec![
        probe(dtype, (Row, Row), 3, AccInit::Zero, Readout::ElemRead),
        probe(
            dtype,
            (Col, Col),
            2,
            AccInit::LoadLds(Col),
            Readout::ElemRead,
        ),
        probe(
            dtype,
            (Row, Col),
            2,
            AccInit::LoadLds(Row),
            Readout::ElemScale,
        ),
    ]
}

fn label(spec: &FragmentProbe) -> String {
    format!(
        "{:?} A {:?} B {:?} K{} init {:?} readout {:?}",
        spec.dtype, spec.a_major, spec.b_major, spec.k_tiles, spec.acc_init, spec.readout
    )
}

fn kernel_name(index: usize) -> String {
    format!("fragment_probe_{index}")
}

fn work_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---- compile rows (toolchain only) ----------------------------------------------------------------------

#[test]
fn every_probe_verifies() {
    for dtype in [WmmaDtype::F16, WmmaDtype::Bf16] {
        for spec in portable_probes(dtype).iter().chain(&declared_probes(dtype)) {
            spec.body("k")
                .verify()
                .unwrap_or_else(|e| panic!("{}: {e}", label(spec)));
        }
    }
}

/// The SPIR-V module for every portable f16 probe is valid Vulkan 1.3: a cooperative-matrix load from
/// workgroup storage at a base, in both layouts, and the accumulator load and store.
#[test]
fn spirv_probes_validate() {
    if !have("llc") || !have("spirv-val") {
        poot_test_util::skip_report("poot-kernelgen", &"llc or spirv-val not on PATH");
        return;
    }
    let dir = work_dir("poot-fragment-probe-spirv");
    for (i, spec) in portable_probes(WmmaDtype::F16).iter().enumerate() {
        let out = artifact_path(&dir, &kernel_name(i), Target::SpirvVulkan);
        compile(&spec.body(&kernel_name(i)), Target::SpirvVulkan, &out)
            .unwrap_or_else(|e| panic!("{}: {e}", label(spec)));
        let v = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            v.status.success(),
            "{}: spirv-val:\n{}",
            label(spec),
            String::from_utf8_lossy(&v.stderr)
        );
    }
}

/// SPIR-V bf16 and the declared element access are typed refusals, not wrong code.
#[test]
fn spirv_refuses_bf16_and_declared_element_access() {
    let dir = work_dir("poot-fragment-probe-spirv-refusals");
    let out = artifact_path(&dir, "refused", Target::SpirvVulkan);
    let bf16 = portable_probes(WmmaDtype::Bf16).remove(0).body("refused");
    let error = compile(&bf16, Target::SpirvVulkan, &out)
        .unwrap_err()
        .to_string();
    assert!(error.to_lowercase().contains("bf16"), "{error}");
    let element = declared_probes(WmmaDtype::F16).remove(0).body("refused");
    let error = compile(&element, Target::SpirvVulkan, &out)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("WmmaElemRead needs a declared accumulator layout"),
        "{error}"
    );
}

/// NVPTX: the opaque layout refuses element access (typed) and lowers the layouts to the matching `wmma`
/// forms: a column operand loads with `.col` and the multiply is specialised on both layouts.
#[test]
fn nvptx_probes_lower_layouts_and_refuse_element_access() {
    if !have("llc") {
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
        return;
    }
    let dir = work_dir("poot-fragment-probe-nvptx");
    let element = declared_probes(WmmaDtype::F16).remove(0).body("refused");
    let error = compile(&element, Target::Nvptx, &dir.join("refused.ptx"))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("WmmaElemRead needs a declared accumulator layout"),
        "{error}"
    );
    for dtype in [WmmaDtype::F16, WmmaDtype::Bf16] {
        for (i, spec) in portable_probes(dtype).iter().enumerate() {
            let out = artifact_path(&dir, &kernel_name(i), Target::Nvptx);
            compile(&spec.body(&kernel_name(i)), Target::Nvptx, &out)
                .unwrap_or_else(|e| panic!("{}: {e}", label(spec)));
            let ptx = std::fs::read_to_string(&out).unwrap();
            let layout = |m: FragmentMajor| if m == Col { "col" } else { "row" };
            let (a, b) = (layout(spec.a_major), layout(spec.b_major));
            // LDS operands reach the generic-address `wmma` forms through `cvta.shared`.
            assert!(
                ptx.contains("cvta.shared"),
                "{}: LDS tile pointer",
                label(spec)
            );
            assert!(
                ptx.contains(&format!("wmma.load.a.sync.aligned.{a}.m16n16k16")),
                "{}: A load .{a} from shared:\n{ptx}",
                label(spec)
            );
            assert!(
                ptx.contains(&format!("wmma.load.b.sync.aligned.{b}.m16n16k16")),
                "{}: B load .{b} from shared",
                label(spec)
            );
            assert!(
                ptx.contains(&format!("wmma.mma.sync.aligned.{a}.{b}.m16n16k16")),
                "{}: multiply .{a}.{b}",
                label(spec)
            );
            if let AccInit::LoadLds(major) = spec.acc_init {
                let c = layout(major);
                assert!(
                    ptx.contains(&format!("wmma.load.c.sync.aligned.{c}.m16n16k16")),
                    "{}: accumulator load .{c}",
                    label(spec)
                );
            }
            if let Readout::StoreLds(major) = spec.readout {
                let d = layout(major);
                assert!(
                    ptx.contains(&format!("wmma.store.d.sync.aligned.{d}.m16n16k16")),
                    "{}: accumulator store .{d}",
                    label(spec)
                );
            }
        }
    }
}

/// AMDGCN lowers every probe, both dtypes, including the declared element access.
#[test]
fn amdgcn_probes_compile() {
    let arch = poot_target::AmdArch::from_isa_name("gfx1151", 32).expect("gfx1151 is gfx11 wave32");
    if std::env::var_os("POOT_AMD_CLANG").is_none() {
        poot_test_util::skip_report("poot-kernelgen", &"POOT_AMD_CLANG unset (no AMD toolchain)");
        return;
    }
    let dir = work_dir("poot-fragment-probe-amdgcn");
    let target = Target::AmdGcn(arch);
    for dtype in [WmmaDtype::F16, WmmaDtype::Bf16] {
        for (i, spec) in portable_probes(dtype)
            .iter()
            .chain(&declared_probes(dtype))
            .enumerate()
        {
            let out = artifact_path(&dir, &kernel_name(i), target);
            compile(&spec.body(&kernel_name(i)), target, &out)
                .unwrap_or_else(|e| panic!("{}: {e}", label(spec)));
        }
    }
}

// ---- device rows ----------------------------------------------------------------------------------------

fn storage(dtype: WmmaDtype) -> BufferStorage {
    match dtype {
        WmmaDtype::F16 => BufferStorage::f16(),
        WmmaDtype::Bf16 => BufferStorage::bf16(),
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Dispatch every portable f16 probe on the wgpu cooperative matrix (RADV).
#[test]
fn fragment_probes_match_the_host_reference_on_wgpu() {
    use poot_wgpu_runtime::{Context, DeviceBuffer};
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            poot_test_util::skip_report("poot-kernelgen", &format_args!("no wgpu context: {e}"));
            return;
        }
    };
    if !have("llc") {
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
        return;
    }
    let upload = |role, storage, bytes: &[u8]| -> DeviceBuffer {
        let buf = ctx
            .alloc_storage(role, storage, bytes.len() / storage.element().byte_width())
            .expect("alloc");
        ctx.write_with(&buf, 0, bytes.len(), &mut |range, out| {
            out.copy_from_slice(&bytes[range]);
        })
        .expect("write");
        buf
    };
    let dir = work_dir("poot-fragment-probe-wgpu");
    for (i, spec) in portable_probes(WmmaDtype::F16).iter().enumerate() {
        let case = Case::new(spec);
        let body = spec.body(&kernel_name(i));
        let out = artifact_path(&dir, &kernel_name(i), Target::SpirvVulkan);
        let artifact = compile_kernel(&body, poot_target::Backend::SpirvVulkan, &out)
            .unwrap_or_else(|e| panic!("{}: {e}", label(spec)));
        let kernel = kernel_handle(&body, &artifact).unwrap();
        let st = storage(spec.dtype);
        let a = upload(BufferRole::Input, st, &Case::operand_bytes(spec, &case.a));
        let b = upload(BufferRole::Input, st, &Case::operand_bytes(spec, &case.b));
        let c0 = upload(
            BufferRole::Input,
            BufferStorage::f32(),
            &f32_bytes(&case.c0),
        );
        let result = ctx
            .alloc_storage(BufferRole::Output, BufferStorage::f32(), 256)
            .expect("alloc");
        ctx.dispatch_dev(
            &kernel_name(i),
            &kernel,
            body.workgroup_size,
            [32, 1, 1],
            &[&a, &b, &c0],
            &result,
        )
        .unwrap_or_else(|e| panic!("{}: dispatch: {e}", label(spec)));
        let mut bytes = vec![0u8; 256 * 4];
        ctx.read_bytes(&result, &mut bytes).unwrap();
        case.check(&label(spec), &f32s(&bytes));
    }
}

/// The same probes through the raw Vulkan runtime.
#[test]
fn fragment_probes_match_the_host_reference_on_vulkan() {
    use poot_vulkan_runtime::{Context, DeviceBuffer};
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            poot_test_util::skip_report("poot-kernelgen", &format_args!("no Vulkan device: {e}"));
            return;
        }
    };
    if !have("llc") {
        poot_test_util::skip_report("poot-kernelgen", &"llc not on PATH");
        return;
    }
    let upload = |role, storage, bytes: &[u8]| -> DeviceBuffer {
        let buf = ctx
            .alloc_storage(role, storage, bytes.len() / storage.element().byte_width())
            .expect("alloc");
        buf.write_with(0, bytes.len(), &mut |range, out| {
            out.copy_from_slice(&bytes[range]);
        })
        .expect("write");
        buf
    };
    let dir = work_dir("poot-fragment-probe-vulkan");
    for (i, spec) in portable_probes(WmmaDtype::F16).iter().enumerate() {
        let case = Case::new(spec);
        let body = spec.body(&kernel_name(i));
        let out = artifact_path(&dir, &kernel_name(i), Target::SpirvVulkan);
        let artifact = compile_kernel(&body, poot_target::Backend::SpirvVulkan, &out)
            .unwrap_or_else(|e| panic!("{}: {e}", label(spec)));
        let kernel = kernel_handle(&body, &artifact).unwrap();
        let st = storage(spec.dtype);
        let a = upload(BufferRole::Input, st, &Case::operand_bytes(spec, &case.a));
        let b = upload(BufferRole::Input, st, &Case::operand_bytes(spec, &case.b));
        let c0 = upload(
            BufferRole::Input,
            BufferStorage::f32(),
            &f32_bytes(&case.c0),
        );
        let result = ctx
            .alloc_storage(BufferRole::Output, BufferStorage::f32(), 256)
            .expect("alloc");
        ctx.dispatch(
            &kernel,
            body.workgroup_size,
            [32, 1, 1],
            &[&a, &b, &c0, &result],
        )
        .unwrap_or_else(|e| panic!("{}: dispatch: {e}", label(spec)));
        let mut bytes = vec![0u8; 256 * 4];
        result.read_bytes(&mut bytes).unwrap();
        case.check(&label(spec), &f32s(&bytes));
    }
}

/// Every portable probe and the declared-map probes, bf16 and f16, on the ROCm gfx11 WMMA.
#[test]
fn fragment_probes_match_the_host_reference_on_rocm() {
    use poot_rocm_runtime::{RocmContext, RocmError};
    let ctx = match RocmContext::new() {
        Ok(ctx) => ctx,
        Err(RocmError::LibraryNotFound(_) | RocmError::NoGpuAgent) => {
            poot_test_util::skip_report("poot-kernelgen", &"no ROCm device");
            return;
        }
        Err(error) => panic!("{error}"),
    };
    let arch = poot_target::AmdArch::from_isa_name(ctx.isa_name(), ctx.wavefront())
        .expect("classify the device arch");
    let target = Target::AmdGcn(arch);
    let upload = |storage: BufferStorage, bytes: &[u8]| {
        let buf = ctx
            .allocate(
                BufferRole::Input,
                storage,
                bytes.len() / storage.element().byte_width(),
            )
            .expect("alloc");
        ctx.write_with(&buf, 0, bytes.len(), &mut |range, out| {
            out.copy_from_slice(&bytes[range]);
        })
        .expect("write");
        buf
    };
    let dir = work_dir("poot-fragment-probe-rocm");
    let mut index = 0;
    for dtype in [WmmaDtype::F16, WmmaDtype::Bf16] {
        for spec in portable_probes(dtype).iter().chain(&declared_probes(dtype)) {
            index += 1;
            let name = kernel_name(index);
            let case = Case::new(spec);
            let body = spec.body(&name);
            let out = artifact_path(&dir, &name, target);
            let artifact = compile_kernel(&body, poot_target::Backend::AmdGcn(arch), &out)
                .unwrap_or_else(|e| panic!("{}: {e}", label(spec)));
            let compiled = kernel_handle(&body, &artifact).unwrap();
            let module = ctx.load_hsaco(&compiled).unwrap();
            let kernel = ctx.lookup_kernel(&module, compiled.entry_point()).unwrap();
            let st = storage(dtype);
            let a = upload(st, &Case::operand_bytes(spec, &case.a));
            let b = upload(st, &Case::operand_bytes(spec, &case.b));
            let c0 = upload(BufferStorage::f32(), &f32_bytes(&case.c0));
            let result = ctx.allocate_f32(256, BufferRole::Output).unwrap();
            let buffers = [&a, &b, &c0, &result];
            let slots: Vec<_> = buffers
                .iter()
                .map(|buffer| (*buffer, buffer.elem_count() as u64))
                .collect();
            let elements = [
                st.element(),
                st.element(),
                BufferStorage::f32().element(),
                BufferStorage::f32().element(),
            ];
            let kernarg = ctx.bind_kernarg(&kernel, &slots).unwrap();
            ctx.dispatch(&kernel, &kernarg, &elements, (32, 1, 1), (32, 1, 1))
                .unwrap_or_else(|e| panic!("{}: dispatch: {e}", label(spec)));
            let mut got = vec![0.0f32; 256];
            ctx.download_f32(&result, &mut got).unwrap();
            case.check(&label(spec), &got);
        }
    }
}

/// Every portable probe, bf16 and f16, on the NVPTX `wmma` path. Runs on a pod (skips without an NVIDIA device).
#[test]
fn fragment_probes_match_the_host_reference_on_ptx() {
    use poot_ptx_runtime::PtxContext;
    let Ok(ctx) = PtxContext::new() else {
        poot_test_util::skip_report("poot-kernelgen", &"no PTX device");
        return;
    };
    let upload = |storage: BufferStorage, bytes: &[u8]| {
        let buf = ctx
            .alloc_storage(
                BufferRole::Input,
                storage,
                bytes.len() / storage.element().byte_width(),
            )
            .expect("alloc");
        ctx.write_with(&buf, 0, bytes.len(), &mut |range, out| {
            out.copy_from_slice(&bytes[range]);
        })
        .expect("write");
        buf
    };
    let dir = work_dir("poot-fragment-probe-ptx");
    let mut index = 0;
    for dtype in [WmmaDtype::F16, WmmaDtype::Bf16] {
        for spec in &portable_probes(dtype) {
            index += 1;
            let name = kernel_name(index);
            let case = Case::new(spec);
            let body = spec.body(&name);
            let out = artifact_path(&dir, &name, Target::Nvptx);
            let artifact = compile_kernel(&body, poot_target::Backend::Nvptx, &out)
                .unwrap_or_else(|e| panic!("{}: {e}", label(spec)));
            let kernel = kernel_handle(&body, &artifact).unwrap();
            let st = storage(dtype);
            let a = upload(st, &Case::operand_bytes(spec, &case.a));
            let b = upload(st, &Case::operand_bytes(spec, &case.b));
            let c0 = upload(BufferStorage::f32(), &f32_bytes(&case.c0));
            let result = ctx.alloc_f32(256).unwrap();
            ctx.dispatch_dev(
                &name,
                &kernel,
                body.workgroup_size,
                body.workgroup_size,
                &[&a, &b, &c0],
                &[a.elem_count(), b.elem_count(), c0.elem_count()],
                &result,
                result.elem_count(),
            )
            .unwrap_or_else(|e| panic!("{}: dispatch: {e}", label(spec)));
            case.check(&label(spec), &ctx.download_f32(&result).unwrap());
        }
    }
}
