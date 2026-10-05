//! Card 559 (R468-016): the two importer-capability probes shipped as committed assets - `rowsum_for`
//! (a fixed-bound `for` loop) and `rmsnorm_dyn` (a runtime-value loop bound read from a metadata
//! buffer) - loaded through `ImportedKernel::{RowsumFor,RmsnormDyn}` and dispatched on wgpu through the
//! public `Device` trait, the same low-level path `executor_contract.rs`'s `assert_trap` probe uses.
//! `pootc/tests/import_run.rs` separately proves pootc can import each kernel fresh from source
//! (`rowsum_for_loop_kernel_imported_from_mir_runs_on_gpu`, `rmsnorm_dyn_...`); this proves the
//! committed asset itself is the one every other crate would load.

use poot_executor::{Arg, BufferRole, Device};
use poot_graph_plan::ImportedKernel;
use poot_runtime_common::DeviceBackend;
use poot_target::BufferStorage;

use crate::device::WgpuDevice;

fn wgpu() -> Option<WgpuDevice> {
    poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
}

/// Compile `body` for SpirvVulkan and load it into `device` under `name`.
fn load(
    device: &mut WgpuDevice,
    name: &str,
    body: &poot_kernel_ir::Body,
) -> <WgpuDevice as Device>::Kernel {
    let dir = std::env::temp_dir().join("poot-gpu-imported-kernel-probe");
    std::fs::create_dir_all(&dir).unwrap();
    let out_path = poot_codegen::artifact_path(&dir, name, poot_codegen::Target::SpirvVulkan);
    poot_codegen::compile(body, poot_codegen::Target::SpirvVulkan, &out_path)
        .unwrap_or_else(|e| panic!("{name} must compile to SPIR-V: {e}"));
    let spv = std::fs::read(&out_path).unwrap();
    let compiled = poot_codegen::kernel_handle(body, poot_codegen::Target::SpirvVulkan, spv);
    device
        .load_kernel(name, compiled)
        .unwrap_or_else(|e| panic!("load {name}: {e:?}"))
}

#[test]
fn rowsum_for_asset_matches_cpu_on_wgpu() {
    let Some(mut device) = wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let body = ImportedKernel::RowsumFor.body().clone();
    let kernel = load(&mut device, "rowsum_for_probe", &body);

    const COLS: usize = 4; // matches the kernel's own baked COLS constant.
    let rows = 5usize;
    let x: Vec<f32> = (0..rows * COLS).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let want: Vec<f32> = (0..rows)
        .map(|r| x[r * COLS..r * COLS + COLS].iter().sum())
        .collect();

    let x_buf = device
        .allocate(BufferRole::Input, BufferStorage::f32(), x.len())
        .unwrap();
    device.write(&x_buf, bytemuck::cast_slice(&x)).unwrap();
    let out_buf = device
        .allocate(BufferRole::Output, BufferStorage::f32(), rows)
        .unwrap();

    device.begin(poot_graph_plan::Submission::Replay).unwrap();
    device
        .dispatch(poot_executor::Dispatch {
            kernel: &kernel,
            inputs: &[Arg {
                buffer: &x_buf,
                elems: x.len() as u32,
            }],
            output: Arg {
                buffer: &out_buf,
                elems: rows as u32,
            },
            threads: [rows as u32, 1, 1],
            workgroup: [64, 1, 1],
            work: rows as u64,
        })
        .unwrap();
    let recording = device
        .finish()
        .unwrap()
        .expect("Submission::Replay always returns a recording");
    device
        .replay(&recording)
        .expect("replay the rowsum_for dispatch");
    device.synchronize().expect("rowsum_for must not fault");

    let mut got = vec![0u8; rows * 4];
    device.read(&out_buf, &mut got).unwrap();
    let got: Vec<f32> = got
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(
        got, want,
        "committed rowsum_for.kir.json must compute out[r] = sum(x[r*COLS..r*COLS+COLS])"
    );
}

#[test]
fn rmsnorm_dyn_asset_matches_cpu_on_wgpu() {
    let Some(mut device) = wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let body = ImportedKernel::RmsnormDyn.body().clone();
    let kernel = load(&mut device, "rmsnorm_dyn_probe", &body);

    let (rows, d) = (3usize, 6usize);
    let x: Vec<f32> = (0..rows * d).map(|i| (i as f32) * 0.3 - 0.9).collect();
    let w: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32) * 0.1).collect();
    let want: Vec<f32> = (0..rows)
        .flat_map(|r| {
            let row = &x[r * d..r * d + d];
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / (d as f32);
            let scale = 1.0 / (mean_sq + 1e-5).sqrt();
            row.iter()
                .zip(&w)
                .map(move |(v, wi)| v * scale * wi)
                .collect::<Vec<_>>()
        })
        .collect();

    let x_buf = device
        .allocate(BufferRole::Input, BufferStorage::f32(), x.len())
        .unwrap();
    device.write(&x_buf, bytemuck::cast_slice(&x)).unwrap();
    let w_buf = device
        .allocate(BufferRole::Input, BufferStorage::f32(), w.len())
        .unwrap();
    device.write(&w_buf, bytemuck::cast_slice(&w)).unwrap();
    let dims = [d as u32];
    let dims_storage = BufferStorage::dense(
        poot_target::ElementKind::I32,
        poot_target::LogicalDType::RawBytes,
    );
    let dims_buf = device
        .allocate(BufferRole::Meta, dims_storage, dims.len())
        .unwrap();
    device
        .write(&dims_buf, bytemuck::cast_slice(&dims))
        .unwrap();
    let out_buf = device
        .allocate(BufferRole::Output, BufferStorage::f32(), rows * d)
        .unwrap();

    device.begin(poot_graph_plan::Submission::Replay).unwrap();
    device
        .dispatch(poot_executor::Dispatch {
            kernel: &kernel,
            inputs: &[
                Arg {
                    buffer: &x_buf,
                    elems: x.len() as u32,
                },
                Arg {
                    buffer: &w_buf,
                    elems: w.len() as u32,
                },
                Arg {
                    buffer: &dims_buf,
                    elems: dims.len() as u32,
                },
            ],
            output: Arg {
                buffer: &out_buf,
                elems: (rows * d) as u32,
            },
            threads: [rows as u32, 1, 1],
            workgroup: [64, 1, 1],
            work: rows as u64,
        })
        .unwrap();
    let recording = device
        .finish()
        .unwrap()
        .expect("Submission::Replay always returns a recording");
    device
        .replay(&recording)
        .expect("replay the rmsnorm_dyn dispatch");
    device.synchronize().expect("rmsnorm_dyn must not fault");

    let mut got = vec![0u8; rows * d * 4];
    device.read(&out_buf, &mut got).unwrap();
    let got: Vec<f32> = got
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-4, "rmsnorm_dyn[{i}]: {g} vs CPU ref {w}");
    }
}
