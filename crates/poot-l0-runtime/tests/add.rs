//! Intel-device row for `Target::SpirvOpenCl`: the generated `add` kernel, compiled to OpenCL SPIR-V by the real
//! `compile()` path, runs on a Level Zero GPU through `poot-l0-runtime` and equals the CPU oracle (the
//! kernel-IR interpreter) bit for bit.
//!
//! `#[ignore]`d: it needs an Intel GPU, the Level Zero loader and the compute runtime on the library path,
//! none of which the model-free lane has. Run it by name on the Lunar Lake laptop with `just test-device-l0`.

use poot_codegen::{Target, compile};
use poot_kernel_ir::fixtures;
use poot_kernel_ir::interp::{Buffer, run};
use poot_l0_runtime::{KernelArg, L0Buffer, L0Context, L0DeviceKind, L0Error};

const N: usize = 4096 + 37;

#[test]
#[ignore = "needs an Intel GPU with the Level Zero loader and compute runtime (the Lunar Lake laptop)"]
fn spirv_opencl_add_matches_cpu_oracle_on_level_zero() {
    let context = L0Context::open(L0DeviceKind::Gpu).expect("a Level Zero GPU context");
    assert_eq!(
        context.vendor_id(),
        0x8086,
        "the Lunar Lake laptop's device is an Intel one"
    );

    // The kernel under test: the real `compile()` product, not a hand-written module.
    let body = fixtures::add_kernel();
    let dir = std::env::temp_dir().join("poot-spirv-opencl-level-zero");
    std::fs::create_dir_all(&dir).unwrap();
    let spv_path = dir.join(format!("{}.spv", body.name));
    compile(&body, Target::SpirvOpenCl, &spv_path).expect("add compiles for SpirvOpenCl");
    let module = context
        .load_module(&std::fs::read(&spv_path).unwrap())
        .expect("the driver accepts the SPIR-V");
    let mut kernel = module
        .kernel(&body.name, body.workgroup_size)
        .expect("the module has the kernel");

    let lhs: Vec<f32> = (0..N).map(|i| i as f32 * 0.37 - 11.0).collect();
    let rhs: Vec<f32> = (0..N).map(|i| (i as f32).sqrt() + 0.001).collect();
    let a = context.alloc_f32(N).unwrap();
    let b = context.alloc_f32(N).unwrap();
    let c = context.alloc_f32(N).unwrap();
    a.upload_f32(&lhs).unwrap();
    b.upload_f32(&rhs).unwrap();
    c.upload_f32(&vec![f32::NAN; N]).unwrap();

    // The kernel ABI: a `ptr addrspace(1)` and an `i64` length per slice, in parameter order.
    let len = N as u64;
    let args = [
        KernelArg::Buffer(&a),
        KernelArg::U64(len),
        KernelArg::Buffer(&b),
        KernelArg::U64(len),
        KernelArg::Buffer(&c),
        KernelArg::U64(len),
    ];
    let workgroups = [N.div_ceil(body.workgroup_size[0] as usize) as u32, 1, 1];
    context.dispatch(&mut kernel, workgroups, &args).unwrap();
    let mut device_out = vec![0.0f32; N];
    c.download_f32(&mut device_out).unwrap();

    // The oracle: the same body on the kernel-IR interpreter.
    let mut buffers = [
        Buffer::from_f32s(&lhs),
        Buffer::from_f32s(&rhs),
        Buffer::from_f32s(&vec![f32::NAN; N]),
    ];
    run(&body, workgroups, &mut buffers).expect("the interpreter runs add");
    let oracle = buffers[2].to_f32s().unwrap();
    let bad: Vec<usize> = (0..N)
        .filter(|&i| device_out[i].to_bits() != oracle[i].to_bits())
        .collect();
    assert!(
        bad.is_empty(),
        "{} of {N} elements differ from the CPU oracle; first {:?}: device {} vs oracle {}",
        bad.len(),
        bad[0],
        device_out[bad[0]],
        oracle[bad[0]]
    );
}

/// The runtime's typed refusals, on the production entry points of a live context: each one is raised before
/// the driver is asked to do the wrong thing.
#[test]
#[ignore = "needs an Intel GPU with the Level Zero loader and compute runtime (the Lunar Lake laptop)"]
fn level_zero_refuses_mismatched_transfers_foreign_resources_and_ragged_spirv() {
    let context = L0Context::open(L0DeviceKind::Gpu).expect("a Level Zero GPU context");
    let buffer = context.alloc_f32(8).unwrap();
    assert!(matches!(
        buffer.upload_f32(&[0.0; 7]),
        Err(L0Error::TransferLength {
            host: 28,
            device: 32
        })
    ));
    assert!(matches!(
        buffer.download_f32(&mut [0.0; 9]),
        Err(L0Error::TransferLength {
            host: 36,
            device: 32
        })
    ));
    assert!(matches!(
        context.load_module(&[0x03, 0x02, 0x23, 0x07, 0x00]),
        Err(L0Error::MisalignedSpirv { len: 5 })
    ));

    let body = fixtures::add_kernel();
    let dir = std::env::temp_dir().join("poot-spirv-opencl-level-zero");
    std::fs::create_dir_all(&dir).unwrap();
    let spv_path = dir.join(format!("{}-refusals.spv", body.name));
    compile(&body, Target::SpirvOpenCl, &spv_path).expect("add compiles for SpirvOpenCl");
    let module = context
        .load_module(&std::fs::read(&spv_path).unwrap())
        .unwrap();
    let mut kernel = module.kernel(&body.name, body.workgroup_size).unwrap();
    let other = L0Context::open(L0DeviceKind::Gpu).expect("a second context on the same device");
    let foreign = other.alloc_f32(8).unwrap();
    let own = context.alloc_f32(8).unwrap();
    // The add kernel has six parameters: arity is checked before any argument is bound.
    assert!(matches!(
        context.dispatch(&mut kernel, [1, 1, 1], &[KernelArg::Buffer(&own)]),
        Err(L0Error::ArgumentCount {
            expected: 6,
            given: 1
        })
    ));
    assert!(matches!(
        context.dispatch(&mut kernel, [1, 1, 1], &six(&foreign, &own)),
        Err(L0Error::ForeignResource { what: "buffer" })
    ));
    assert!(matches!(
        other.dispatch(&mut kernel, [1, 1, 1], &[]),
        Err(L0Error::ForeignResource { what: "kernel" })
    ));
}

/// The add kernel's six arguments (a buffer and a length per slice) with `first` as its first buffer and `rest`
/// as the other two.
fn six<'a>(first: &'a L0Buffer<'a>, rest: &'a L0Buffer<'a>) -> [KernelArg<'a>; 6] {
    [
        KernelArg::Buffer(first),
        KernelArg::U64(8),
        KernelArg::Buffer(rest),
        KernelArg::U64(8),
        KernelArg::Buffer(rest),
        KernelArg::U64(8),
    ]
}
