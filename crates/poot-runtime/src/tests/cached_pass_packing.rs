//! Ordered cached dispatches exercise wgpu's per-dispatch resource tracking inside one pass.
//! The Card 552 typed device-timing path's forced per-dispatch attribution is the control, never a
//! timing baseline.

use crate::{
    BufferRole, BufferStorage, CachedDispatch, CompiledKernel, Context, DeviceBuffer, RuntimeError,
};
use poot_kernel_ir::{Body, fixtures};
use poot_runtime_common::DeviceBackend;
use poot_test_util::device_skip::open_or_skip;
use poot_test_util::kernel_fixtures::square_kernel;

fn kernel(body: &Body, name: &str) -> CompiledKernel {
    let dir = std::env::temp_dir().join("poot-runtime-cached-pass-packing");
    std::fs::create_dir_all(&dir).unwrap();
    let path = poot_codegen::artifact_path(&dir, name, poot_codegen::Target::SpirvVulkan);
    poot_codegen::compile(body, poot_codegen::Target::SpirvVulkan, &path).unwrap();
    poot_codegen::kernel_handle(
        body,
        poot_codegen::Target::SpirvVulkan,
        std::fs::read(path).unwrap(),
    )
}

fn cached(
    ctx: &Context,
    label: &str,
    key: &str,
    kernel: &CompiledKernel,
    inputs: &[&DeviceBuffer],
    output: &DeviceBuffer,
    threads: u32,
) -> CachedDispatch {
    let input_lens: Vec<u32> = inputs.iter().map(|b| b.elem_count()).collect();
    ctx.build_cached_dispatch(
        label,
        key,
        kernel,
        [64, 1, 1],
        [threads, 1, 1],
        inputs,
        &input_lens,
        output,
        output.elem_count(),
    )
    .unwrap()
}

fn arrays_equal(actual: &[Vec<f32>], expected: &[Vec<f32>], stage: &str) {
    assert_eq!(actual.len(), expected.len());
    for (buffer, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.len(), e.len());
        for (lane, (&a, &e)) in a.iter().zip(e).enumerate() {
            assert!(a.is_finite(), "{stage} buffer {buffer} lane {lane}: {a}");
            assert_eq!(a, e, "{stage} buffer {buffer} lane {lane}");
        }
    }
}

fn dependency_chain(ctx: &Context, passes: usize) -> Vec<Vec<f32>> {
    let add = kernel(&fixtures::add_kernel(), "packing-add");
    let square = kernel(&square_kernel(), "packing-square");
    let inputs: Vec<_> = (0..4)
        .map(|_| ctx.alloc_storage(BufferRole::Input, BufferStorage::f32(), 129))
        .collect();
    // X, T (short), Y, Z, O. Whole-buffer clone aliases retain the same native identity.
    let outputs: Vec<_> = [129, 5, 129, 129, 129]
        .map(|n| ctx.alloc_storage(BufferRole::Output, BufferStorage::f32(), n))
        .into();
    let alias = outputs[0].clone();
    assert_eq!(alias.id(), outputs[0].id());
    let dispatches = vec![
        cached(
            ctx,
            "packing-0",
            "packing-add",
            &add,
            &[&inputs[0], &inputs[1]],
            &outputs[0],
            129,
        ),
        cached(
            ctx,
            "packing-1",
            "packing-add",
            &add,
            &[&inputs[2], &inputs[3]],
            &outputs[1],
            5,
        ),
        cached(
            ctx,
            "packing-2",
            "packing-square",
            &square,
            &[&alias],
            &outputs[2],
            129,
        ),
        cached(
            ctx,
            "packing-3",
            "packing-add",
            &add,
            &[&inputs[2], &inputs[3]],
            &outputs[0],
            129,
        ),
        cached(
            ctx,
            "packing-4",
            "packing-square",
            &square,
            &[&alias],
            &outputs[3],
            129,
        ),
        cached(
            ctx,
            "packing-5",
            "packing-add",
            &add,
            &[&outputs[2], &outputs[3]],
            &outputs[4],
            129,
        ),
    ];
    assert_eq!(
        dispatches.iter().map(|d| d.groups).collect::<Vec<_>>(),
        [
            [3, 1, 1],
            [1, 1, 1],
            [3, 1, 1],
            [3, 1, 1],
            [3, 1, 1],
            [3, 1, 1]
        ]
    );
    drop(alias);
    let observer = ctx.counters();
    let mut last_expected = Vec::new();
    let mut all_actual = Vec::new();
    for seed in [0, 7, 0] {
        let a: Vec<_> = (0..129).map(|i| (i % 11 + seed) as f32).collect();
        let b: Vec<_> = (0..129).map(|i| (i % 7 + 2) as f32).collect();
        let c: Vec<_> = (0..129).map(|i| (i % 13 + 30 + seed) as f32).collect();
        let d: Vec<_> = (0..129).map(|i| (i % 5 + 9) as f32).collect();
        for (buffer, data) in inputs.iter().zip([&a, &b, &c, &d]) {
            ctx.write_f32(buffer, data).unwrap();
        }
        for (i, output) in outputs.iter().enumerate() {
            ctx.write_f32(
                output,
                &vec![-1000.0 - i as f32; output.elem_count() as usize],
            )
            .unwrap();
        }
        let x: Vec<_> = c.iter().zip(&d).map(|(c, d)| c + d).collect();
        let y: Vec<_> = a.iter().zip(&b).map(|(a, b)| (a + b) * (a + b)).collect();
        let z: Vec<_> = x.iter().map(|x| x * x).collect();
        let o: Vec<_> = y.iter().zip(&z).map(|(y, z)| y + z).collect();
        last_expected = vec![x.clone(), x[..5].to_vec(), y, z, o];
        let before = observer.snapshot();
        ctx.submit_cached(&dispatches.iter().collect::<Vec<_>>())
            .unwrap();
        let after = observer.snapshot();
        assert_eq!(after.compute_passes - before.compute_passes, passes);
        assert_eq!(after.dispatches - before.dispatches, 6);
        assert_eq!(after.submits - before.submits, 1);
        assert_eq!(after.native_submits - before.native_submits, 1);
        assert_eq!(after.waits - before.waits, 0);
        assert_eq!(after.pipeline_builds, before.pipeline_builds);
        assert_eq!(after.bind_group_creates, before.bind_group_creates);
        let actual: Vec<_> = outputs
            .iter()
            .map(|o| ctx.download_f32(o).unwrap())
            .collect();
        arrays_equal(&actual, &last_expected, "dependency chain");
        eprintln!(
            "packing seed={seed}, X[0,64,128]={:?}, O[0,64,128]={:?}",
            [actual[0][0], actual[0][64], actual[0][128]],
            [actual[4][0], actual[4][64], actual[4][128]]
        );
        all_actual.extend(actual);
    }
    // A final submission outlives the caller's cached objects and input handles. wgpu must retain
    // their submitted uses until the ordinary output readback completes. No native pool is touched.
    for (i, output) in outputs.iter().enumerate() {
        ctx.write_f32(
            output,
            &vec![-2000.0 - i as f32; output.elem_count() as usize],
        )
        .unwrap();
    }
    ctx.submit_cached(&dispatches.iter().collect::<Vec<_>>())
        .unwrap();
    drop(dispatches);
    drop(inputs);
    drop(add);
    drop(square);
    let actual: Vec<_> = outputs
        .iter()
        .map(|o| ctx.download_f32(o).unwrap())
        .collect();
    arrays_equal(&actual, &last_expected, "submitted resource lifetime");
    all_actual.extend(actual);
    all_actual
}

#[test]
fn grouped_cached_dependencies_bindings_grids_and_lifetime_gpu() {
    let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
        return;
    };
    let grouped = dependency_chain(&ctx, 1);
    let Some(control) = open_or_skip(DeviceBackend::Wgpu, Context::new_with_device_timing(64))
    else {
        return;
    };
    // Card 552 review N1: forcing per-dispatch attribution now requires real TIMESTAMP_QUERY support
    // (unlike the deleted legacy label-keyed Profiler, which forced one pass per dispatch even
    // without device-time collection). On hardware without TIMESTAMP_QUERY this control degrades to
    // the same 1-pass grouping as `ctx` above.
    let expected_passes = if control.timestamps_available() { 6 } else { 1 };
    let per_dispatch = dependency_chain(&control, expected_passes);
    arrays_equal(&grouped, &per_dispatch, "grouped versus per-dispatch");
}

#[test]
fn grouped_cached_fault_order_reset_and_repeated_delivery_gpu() {
    let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
        return;
    };
    let body: Body = serde_json::from_str(include_str!(
        "../../../poot-rocm-gpu/assets/assert_trap.kir.json"
    ))
    .unwrap();
    assert!(body.has_trap());
    let trap = kernel(&body, "packing-trap");
    let add = kernel(&fixtures::add_kernel(), "packing-fault-add");
    let data = ctx.upload_f32(&[10.0, 20.0, 30.0, 40.0]);
    let idx: Vec<_> = (0..3)
        .map(|_| ctx.upload_u32_writable(&[3, 0, 2, 1]))
        .collect();
    let outputs: Vec<_> = (0..4).map(|_| ctx.alloc_f32(4)).collect();
    let labels = ["packing-early", "packing-middle", "packing-late"];
    let mut dispatches = Vec::new();
    for i in 0..3 {
        dispatches.push(cached(
            &ctx,
            labels[i],
            "packing-trap",
            &trap,
            &[&idx[i], &data],
            &outputs[i],
            4,
        ));
        if i == 0 {
            dispatches.push(cached(
                &ctx,
                "packing-ordinary",
                "packing-fault-add",
                &add,
                &[&data, &data],
                &outputs[3],
                4,
            ));
        }
    }
    for faults in [
        [false, false, false],
        [true, false, false],
        [false, true, false],
        [false, false, true],
        [true, false, true],
        [false, false, false],
    ] {
        for (idx, bad) in idx.iter().zip(faults) {
            ctx.write_u32(idx, &[3, 0, if bad { 4 } else { 2 }, 1])
                .unwrap();
        }
        let before = ctx.counters().snapshot();
        ctx.submit_cached(&dispatches.iter().collect::<Vec<_>>())
            .unwrap();
        let after = ctx.counters().snapshot();
        assert_eq!(after.compute_passes - before.compute_passes, 1);
        assert_eq!(after.dispatches - before.dispatches, 4);
        let registered: Vec<_> = ctx
            .pending_faults
            .borrow()
            .iter()
            .map(|f| f.label.clone())
            .collect();
        eprintln!("registered fault order: {registered:?}");
        let result = ctx.download_f32(&outputs[2]);
        assert!(ctx.pending_faults.borrow().is_empty());
        if let Some(first) = faults.iter().position(|bad| *bad) {
            let err = result.expect_err("a fault must arrive before a successful result");
            let RuntimeError::KernelAssertFailed { kernel, code } = err else {
                panic!("{err:?}");
            };
            assert_eq!(kernel, labels[first]);
            assert_ne!(code, 0);
            eprintln!("faults={faults:?}: {kernel} code={code}");
        } else {
            assert_eq!(result.unwrap(), [40.0, 10.0, 30.0, 20.0]);
            for output in &outputs[..2] {
                assert_eq!(ctx.download_f32(output).unwrap(), [40.0, 10.0, 30.0, 20.0]);
            }
            assert_eq!(
                ctx.download_f32(&outputs[3]).unwrap(),
                [20.0, 40.0, 60.0, 80.0]
            );
        }
    }
}

#[test]
fn grouped_cached_profile_empty_and_singleton_contract_gpu() {
    let add = kernel(&fixtures::add_kernel(), "packing-profile-add");
    let square = kernel(&square_kernel(), "packing-profile-square");
    for mode in 0..3 {
        let opened = if mode == 0 {
            Context::new()
        } else {
            Context::new_with_device_timing(64)
        };
        let Some(mut ctx) = open_or_skip(DeviceBackend::Wgpu, opened) else {
            return;
        };
        if mode == 2 {
            ctx.timestamps = false;
        }
        let timestamps = mode != 0 && ctx.timestamps_available();
        eprintln!("profile mode={mode}, timestamp queries={timestamps}");
        // Empty timestamp-enabled profiling has its pre-existing zero-query validation behavior;
        // the successful unprofiled and host-only empty cases still submit without a pass or wait.
        if !timestamps {
            let before = ctx.counters().snapshot();
            ctx.submit_cached(&[]).unwrap();
            let after = ctx.counters().snapshot();
            assert_eq!(after.compute_passes, before.compute_passes);
            assert_eq!(after.dispatches, before.dispatches);
            assert_eq!(after.submits - before.submits, 1);
            assert_eq!(after.native_submits - before.native_submits, 1);
            assert_eq!(after.waits, before.waits);
        }
        let a = ctx.upload_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = ctx.upload_f32(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let x = ctx.alloc_f32(5);
        let y = ctx.alloc_f32(5);
        let first = cached(
            &ctx,
            "profile-add",
            "packing-profile-add",
            &add,
            &[&a, &b],
            &x,
            5,
        );
        let second = cached(
            &ctx,
            "profile-square",
            "packing-profile-square",
            &square,
            &[&x],
            &y,
            5,
        );
        let before = ctx.counters().snapshot();
        ctx.submit_cached(&[&first, &second]).unwrap();
        let after = ctx.counters().snapshot();
        // Card 552 review N1: device timing without real TIMESTAMP_QUERY support (mode 2) must not
        // perturb grouping either - only `timestamps` (device timing on *and* real adapter support)
        // forces one pass per dispatch, matching `self.device_timing.get() && self.timestamps` in
        // `submit_encoded`.
        assert_eq!(
            after.compute_passes - before.compute_passes,
            if timestamps { 2 } else { 1 }
        );
        assert_eq!(after.dispatches - before.dispatches, 2);
        assert_eq!(after.submits - before.submits, 1);
        assert_eq!(after.native_submits - before.native_submits, 1);
        // Card 552: unlike the deleted legacy label-keyed Profiler (which synced immediately inside
        // the submit to read back per-dispatch device time), the typed device-timing path registers
        // its map without polling (review F1/F3) - `submit_cached` itself never waits, timestamps or
        // not; only an explicit `drain_device_timing` (not called here) or a later capacity drain
        // would.
        assert_eq!(after.waits, before.waits);
        assert_eq!(
            ctx.download_f32(&x).unwrap(),
            [11.0, 22.0, 33.0, 44.0, 55.0]
        );
        assert_eq!(
            ctx.download_f32(&y).unwrap(),
            [121.0, 484.0, 1089.0, 1936.0, 3025.0]
        );
        let before = ctx.counters().snapshot();
        ctx.submit_cached(&[&first]).unwrap();
        let after = ctx.counters().snapshot();
        assert_eq!(after.compute_passes - before.compute_passes, 1);
        assert_eq!(after.dispatches - before.dispatches, 1);
        assert_eq!(after.waits, before.waits);
        assert_eq!(
            ctx.download_f32(&x).unwrap(),
            [11.0, 22.0, 33.0, 44.0, 55.0]
        );
    }
}

#[test]
fn grouped_cached_invalid_late_grid_prevents_successful_return_gpu() {
    let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
        return;
    };
    let add = kernel(&fixtures::add_kernel(), "packing-invalid-add");
    let a = ctx.upload_f32(&[1.0; 5]);
    let b = ctx.upload_f32(&[2.0; 5]);
    let x = ctx.alloc_f32(5);
    let y = ctx.alloc_f32(5);
    let first = cached(
        &ctx,
        "valid-prefix",
        "packing-invalid-add",
        &add,
        &[&a, &b],
        &x,
        5,
    );
    let mut last = cached(
        &ctx,
        "invalid-late",
        "packing-invalid-add",
        &add,
        &[&x, &b],
        &y,
        5,
    );
    last.groups[0] = ctx.max_workgroups() + 1;
    let before = ctx.counters().snapshot();
    let mut returned_success = false;
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ctx.submit_cached(&[&first, &last]).unwrap();
        returned_success = true;
    }));
    assert!(
        failed.is_err(),
        "native validation must reject the later malformed grid"
    );
    assert!(
        !returned_success,
        "no successful return can publish a token/state"
    );
    assert_eq!(
        ctx.counters().snapshot().submits,
        before.submits,
        "the one encoder fails validation before the queue-submit tail completes"
    );
    // This preserves the existing uncaptured-validation panic. It makes no rollback claim about
    // bytes of a failed invocation, and creates no artificial split or prefix submission.
}
