use super::{RuntimeError, segment_byte_layout};

#[test]
fn flush_uploads_completes_pending_bytes_without_a_later_submit_gpu() {
    use crate::Context;
    use poot_runtime_common::DeviceBackend;
    use poot_test_util::device_skip::open_or_skip;
    use std::time::Duration;

    let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
        return;
    };
    let a: Vec<u8> = (0..256).map(|i| (i % 127 + 1) as u8).collect();
    let b: Vec<u8> = a.iter().map(|v| v + 128).collect();
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pending-upload-witness"),
        size: a.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Native fixture setup and mapping are outside the helper's counter bracket. This read never
    // submits: a normal Context download would flush a missing upload and hide the omission.
    let read_without_submit = || {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(10)),
            })
            .expect("finite mapping poll");
        rx.recv_timeout(Duration::from_secs(10))
            .expect("mapping callback deadline")
            .expect("mapping succeeded");
        let bytes = buffer.slice(..).get_mapped_range().to_vec();
        buffer.unmap();
        bytes
    };
    ctx.queue.write_buffer(&buffer, 0, &a);
    ctx.queue.submit([]);
    assert_eq!(read_without_submit(), a, "settled nonzero A");

    ctx.queue.write_buffer(&buffer, 0, &b);
    let counters = ctx.counters();
    let before = counters.snapshot();
    ctx.flush_uploads_and_wait().unwrap();
    let after = counters.snapshot();
    let got = read_without_submit();
    eprintln!(
        "pending upload A[0..8]={:?}, B={:?}, got={:?}",
        &a[..8],
        &b[..8],
        &got[..8]
    );
    assert_eq!(
        got, b,
        "every pending B byte must be visible without another submit"
    );
    let mut expected = before;
    expected.native_submits += 1;
    expected.waits += 1;
    assert_eq!(after, expected, "only one native upload submit and wait");
}

mod segment_layout_tests {
    use super::{RuntimeError, segment_byte_layout};

    #[test]
    fn segment_layout_is_contiguous_in_request_order() {
        let (offsets, total) = segment_byte_layout([(2, 2), (0, 5), (3, 8)]).unwrap();
        assert_eq!(offsets, vec![0, 8, 8]);
        assert_eq!(total, 20);
    }

    #[test]
    fn segment_layout_rejects_overlong_segment() {
        assert!(matches!(
            segment_byte_layout([(1, 1), (5, 4)]),
            Err(RuntimeError::SegmentLength {
                segment: 1,
                requested: 5,
                available: 4
            })
        ));
    }
}

/// Card 531c: `dispatch_dev` has no per-dispatch sync of its own, so a fault it raises
/// must still surface at the resident path's next real sync point (`Context::download_f32`), not be
/// silently swallowed the way it was before this fix (confirmed empirically:
/// `dispatch_dev` on a faulting input returned `Ok(())`, `out` held robust-buffer-access garbage).
mod pending_fault_tests {
    use crate::{Context, RuntimeError};
    use poot_kernel_ir::Body;

    /// Compile `poot-rocm-gpu`'s committed `assert_trap.kir.json` fixture (card 531c: `out[i] =
    /// data[idx[i]]`, a real rustc-inserted bounds check, not a hand-built `Body`) to SpirV.
    fn compile_assert_trap_spirv() -> (crate::CompiledKernel, Body) {
        let body: Body = serde_json::from_str(include_str!(
            "../../poot-rocm-gpu/assets/assert_trap.kir.json"
        ))
        .expect("deserialize assert_trap fixture");
        let dir = std::env::temp_dir().join("poot-runtime-pending-fault-probe");
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(
            &dir,
            "assert_trap_pending_fault",
            poot_codegen::Target::SpirvVulkan,
        );
        poot_codegen::compile(&body, poot_codegen::Target::SpirvVulkan, &out)
            .expect("assert_trap must compile to SpirV");
        let bytes = std::fs::read(&out).expect("read compiled spv");
        let kernel = poot_codegen::kernel_handle(&body, poot_codegen::Target::SpirvVulkan, bytes);
        (kernel, body)
    }

    /// GREEN: every `idx[i]` in range, no lane traps, `dispatch_dev` + the following `download_f32` both
    /// succeed and `out[i] == data[idx[i]]`. RED (mutation: `idx[2] = data.len()`, one past the end):
    /// lane 2's bounds check fails; `dispatch_dev` itself still returns `Ok` (it never syncs), but the
    /// following `download_f32` must return `Err(KernelAssertFailed)` instead of `Ok` with garbage.
    /// Mutation that removes the step-sync check (e.g. drop `download_f32`'s call to
    /// `stage_pending_faults`/`check_staged_faults`) makes this red: `download_f32` returns `Ok`.
    #[test]
    fn dispatch_dev_fault_surfaces_at_the_next_download() {
        let (spv, body) = compile_assert_trap_spirv();
        assert!(
            body.has_trap(),
            "assert_trap's data[idx[i]] bounds check must compile to a real Trap"
        );
        let ctx = match Context::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP dispatch_dev_fault_surfaces_at_the_next_download: no GPU ({e})");
                return;
            }
        };

        // Green: every idx in range.
        let idx_ok = ctx.upload_u32(&[3u32, 0, 2, 1]);
        let data = ctx.upload_f32(&[10.0f32, 20.0, 30.0, 40.0]);
        let out_ok = ctx.alloc_f32(4);
        ctx.dispatch_dev(
            "assert_trap_ok",
            &spv,
            [64, 1, 1],
            [4, 1, 1],
            &[&idx_ok, &data],
            &out_ok,
        )
        .expect("dispatch_dev never syncs, so it always returns Ok on its own");
        let got = ctx
            .download_f32(&out_ok)
            .expect("in-range indices must not fault");
        assert_eq!(
            got,
            [40.0, 10.0, 30.0, 20.0],
            "assert_trap must compute out[i] = data[idx[i]] when every index is in range"
        );

        // Red (data, not code). MUTATION (card 531c): idx[2] = data.len() (one past the
        // end) forces lane 2's bounds-check Assert to fail.
        let idx_bad = ctx.upload_u32(&[3u32, 0, 4, 1]);
        let out_bad = ctx.alloc_f32(4);
        ctx.dispatch_dev(
            "assert_trap_fault",
            &spv,
            [64, 1, 1],
            [4, 1, 1],
            &[&idx_bad, &data],
            &out_bad,
        )
        .expect("dispatch_dev never syncs, so it always returns Ok even for a faulting input");
        let err = ctx.download_f32(&out_bad).expect_err(
            "idx[2] == 4 is out of range for a 4-element data buffer; the resident path's next sync must \
             report the fault, not silently return the faulting lane's garbage",
        );
        let RuntimeError::KernelAssertFailed { kernel, code } = err else {
            panic!("expected KernelAssertFailed, got {err:?}");
        };
        assert_eq!(kernel, "assert_trap_fault");
        assert_ne!(
            code, 0,
            "a fired trap's code is never the 0 sentinel (no-fault)"
        );
        eprintln!(
            "card 531c proven: a dispatch_dev fault surfaces at the next download_f32 \
             (code {code})"
        );
    }

    /// `submit_cached`'s error-word buffer is the SAME buffer across every step (unlike
    /// `submit_dispatches`, which rebuilds one fresh per call), so it must be re-zeroed and
    /// re-registered on every call - built once here (`build_cached_dispatch`), then re-submitted 3
    /// times over the SAME `CachedDispatch`/buffers with different index content
    /// (`Context::write_u32`), proving both halves: a step-2 fault is caught, and it does not haunt
    /// step 3 once the input is back in range.
    #[test]
    fn submit_cached_error_word_is_rezeroed_and_rechecked_every_step() {
        let (spv, body) = compile_assert_trap_spirv();
        assert!(body.has_trap());
        let ctx = match Context::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "SKIP submit_cached_error_word_is_rezeroed_and_rechecked_every_step: no GPU ({e})"
                );
                return;
            }
        };

        // Rewritten in place between steps (`Context::write_u32`), so it needs `upload_u32_writable`
        // (COPY_DST), not the plain `upload_u32` `dispatch_dev_fault_surfaces_at_the_next_download` uses.
        let idx = ctx.upload_u32_writable(&[3u32, 0, 2, 1]);
        let data = ctx.upload_f32(&[10.0f32, 20.0, 30.0, 40.0]);
        let out = ctx.alloc_f32(4);
        let cached = ctx
            .build_cached_dispatch(
                "assert_trap_cached",
                "assert_trap_cached_key",
                &spv,
                [64, 1, 1],
                [4, 1, 1],
                &[&idx, &data],
                &[idx.elem_count(), data.elem_count()],
                &out,
                out.elem_count(),
            )
            .expect("build_cached_dispatch");

        // Step 1: clean.
        ctx.submit_cached(&[&cached])
            .expect("submit_cached never syncs on its own");
        let got1 = ctx
            .download_f32(&out)
            .expect("step 1: in-range indices must not fault");
        assert_eq!(got1, [40.0, 10.0, 30.0, 20.0]);

        // Step 2. MUTATION (card 531c): idx[2] = data.len() (one past the end), reusing
        // the SAME CachedDispatch/error buffer as step 1.
        ctx.write_u32(&idx, &[3u32, 0, 4, 1]).unwrap();
        ctx.submit_cached(&[&cached])
            .expect("submit_cached never syncs on its own");
        let pending = ctx.pending_faults.borrow();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].label, "assert_trap_cached");
        let fault_storage = pending[0].storage.clone();
        drop(pending);
        ctx.flush_uploads_and_wait().unwrap();
        let pending = ctx.pending_faults.borrow();
        assert_eq!(
            pending.len(),
            1,
            "upload completion must retain the pending fault"
        );
        assert_eq!(pending[0].label, "assert_trap_cached");
        assert_eq!(pending[0].storage, fault_storage);
        drop(pending);
        let err = ctx.download_f32(&out).expect_err(
            "step 2: idx[2] == 4 is out of range; the reused error buffer must still fault",
        );
        assert!(
            matches!(err, RuntimeError::KernelAssertFailed { ref kernel, code: 2 }
            if kernel == "assert_trap_cached"),
            "expected labeled KernelAssertFailed code 2, got {err:?}"
        );

        // Step 3: back in range, same CachedDispatch. If the error word were not re-zeroed before this
        // step's dispatch, step 2's stale nonzero code would still be sitting there and this would fault
        // again even though nothing is wrong with step 3's input.
        ctx.write_u32(&idx, &[3u32, 0, 2, 1]).unwrap();
        ctx.submit_cached(&[&cached])
            .expect("submit_cached never syncs on its own");
        let got3 = ctx.download_f32(&out).expect(
            "step 3: the error word must be re-zeroed every step; a stale code from step 2 must not \
             resurface once the input is back in range",
        );
        assert_eq!(got3, [40.0, 10.0, 30.0, 20.0]);
        eprintln!(
            "card 531c proven: submit_cached re-zeroes and re-checks its error word \
             every step (clean -> fault -> clean over the same CachedDispatch)"
        );
    }

    /// Card 547a SC-005: a warm decode's direct result/validation staging increments the one memory
    /// service's physical allocation count/bytes under `BufferRole::Staging`, even though
    /// `alloc_f32`/`buffer_allocs()` (the existing `alloc_f32`-only counter) and the dispatch count are
    /// unchanged - these were never `alloc_storage` calls and so were invisible to `MemoryCounters`
    /// before this card. Review F2 widened this from only the length/error-word buffers
    /// `build_cached_dispatch` bakes into its `bind_group` to every direct-allocation producer the
    /// card's Scope names: `submit_cached`'s pending-fault error-word staging (`stage_pending_faults`),
    /// `download_f32`'s and `read_bytes`'s and `download_segments_u32`'s own readback staging buffer -
    /// each call must bump the Staging count again, not just the one at build time.
    ///
    /// Mutation (recorded here, applied by hand, never left in the tree; card 547a): in
    /// `Context::download_f32`/`download_segments_u32`/`read_bytes`
    /// (`crates/poot-runtime/src/context/readback.rs`), commented out each function's
    /// `let _staging_guard = self.tag_alloc(BufferRole::Staging, ...);` line, and in
    /// `Context::stage_pending_faults` (`crates/poot-runtime/src/context/dispatch.rs`), changed `let
    /// guard = self.tag_alloc(BufferRole::Staging, 4);` to charge a fresh, disconnected
    /// `MemoryCounters::new()` instead. Result: RED - `staging_after_download.allocations >
    /// staging_after_build.allocations` failed ("download_f32's own readback staging buffer, plus the
    /// pending-fault error-word staging stage_pending_faults creates, must both be charged: before=1,
    /// after=1"), and the two assertions that follow it (`read_bytes`, `download_segments_u32`) failed
    /// the same way. Reverted: GREEN.
    #[test]
    fn warm_decode_staging_allocation_increments_the_memory_service_independent_of_alloc_f32() {
        let (spv, body) = compile_assert_trap_spirv();
        assert!(body.has_trap());
        let ctx = match Context::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "SKIP warm_decode_staging_allocation_increments_the_memory_service_independent_of_alloc_f32: no GPU ({e})"
                );
                return;
            }
        };
        let idx = ctx.upload_u32_writable(&[3u32, 0, 2, 1]);
        let data = ctx.upload_f32(&[10.0f32, 20.0, 30.0, 40.0]);
        let out = ctx.alloc_f32(4);

        let staging_snapshot = |ctx: &Context| -> poot_runtime_common::MemoryCounterSnapshot {
            ctx.memory()
                .into_iter()
                .find(|&(role, _)| role == poot_runtime_common::BufferRole::Staging)
                .map(|(_, snap)| snap)
                .unwrap_or_default()
        };

        let buffer_allocs_before = ctx.buffer_allocs();
        let staging_before = staging_snapshot(&ctx);

        let cached = ctx
            .build_cached_dispatch(
                "warm_decode_staging",
                "warm_decode_staging_key",
                &spv,
                [64, 1, 1],
                [4, 1, 1],
                &[&idx, &data],
                &[idx.elem_count(), data.elem_count()],
                &out,
                out.elem_count(),
            )
            .expect("build_cached_dispatch");

        let buffer_allocs_after = ctx.buffer_allocs();
        let staging_after_build = staging_snapshot(&ctx);

        assert_eq!(
            buffer_allocs_after, buffer_allocs_before,
            "the length/error-word buffers are not alloc_f32 calls, so buffer_allocs() must be unchanged"
        );
        assert!(
            staging_after_build.allocations > staging_before.allocations,
            "staging allocation count must increase: before={}, after={}",
            staging_before.allocations,
            staging_after_build.allocations
        );
        assert!(
            staging_after_build.live_bytes > staging_before.live_bytes,
            "staging live bytes must increase: before={}, after={}",
            staging_before.live_bytes,
            staging_after_build.live_bytes
        );

        // F2: `submit_cached` registers the error-word buffer as a pending fault on every call (every
        // index here stays in range, so nothing actually traps); the following `download_f32` is the
        // step-boundary sync that drains it through `stage_pending_faults`, charging its own
        // per-fault Staging buffer on top of `download_f32`'s own readback staging buffer.
        ctx.submit_cached(&[&cached])
            .expect("submit_cached never syncs on its own");
        let got = ctx
            .download_f32(&out)
            .expect("in-range indices must not fault");
        assert_eq!(got, [40.0, 10.0, 30.0, 20.0]);
        let staging_after_download = staging_snapshot(&ctx);
        assert!(
            staging_after_download.allocations > staging_after_build.allocations,
            "download_f32's own readback staging buffer, plus the pending-fault error-word staging \
             stage_pending_faults creates, must both be charged: before={}, after={}",
            staging_after_build.allocations,
            staging_after_download.allocations
        );

        // F2: `read_bytes`'s own readback staging buffer.
        let mut raw = [0u8; 16];
        ctx.read_bytes(&out, &mut raw).expect("read_bytes");
        let staging_after_read_bytes = staging_snapshot(&ctx);
        assert!(
            staging_after_read_bytes.allocations > staging_after_download.allocations,
            "read_bytes's own readback staging buffer must be charged: before={}, after={}",
            staging_after_download.allocations,
            staging_after_read_bytes.allocations
        );

        // F2: `download_segments_u32`'s own concatenated readback staging buffer.
        let _ = ctx
            .download_segments_u32(&[crate::BufferSegment {
                buffer: &idx,
                words: 4,
            }])
            .expect("download_segments_u32");
        let staging_after_segments = staging_snapshot(&ctx);
        assert!(
            staging_after_segments.allocations > staging_after_read_bytes.allocations,
            "download_segments_u32's own readback staging buffer must be charged: before={}, after={}",
            staging_after_read_bytes.allocations,
            staging_after_segments.allocations
        );
    }
}

/// Card 527 review G1: `write_f32`/`write_i32`/`write_u32` are release-checked, `Result`-returning copy
/// entry points, not a panic (round 1's mistake) and not silently unchecked (round 2's
/// mistake, this round's fix). Each test drives the real public API - `Context::upload_i32_writable` +
/// `DeviceBuffer::with_storage` + `Context::write_f32` - no forged private field. Needs a real device
/// only to obtain a `wgpu::Buffer` handle; the check itself is host-side bookkeeping. `write_f32_at`/
/// `write_i32_at` (the former partial-write analogues, Card 527 review G2) are deleted (Card 547a,
/// strict dead-pub).
mod storage_checked_write_tests {
    use crate::*;
    use poot_runtime_common::DeviceBackend;
    use poot_test_util::device_skip::open_or_skip;

    /// A write into a buffer whose recorded storage genuinely disagrees (never retagged) is refused.
    ///
    /// Mutation (recorded here, never left in the tree; card 527 review G1): removing
    /// `checked_storage(...)?;` from `write_f32` (`crates/poot-runtime/src/context/transfer.rs`) turns
    /// this test red in a release build: the i32-tagged buffer accepts the f32 write silently instead of
    /// being refused. `cargo nextest run --release -p poot-runtime
    /// write_into_a_genuinely_mismatched_buffer_is_refused` -> FAILED (`expect_err` sees `Ok`); reverted
    /// -> passes.
    #[test]
    fn write_into_a_genuinely_mismatched_buffer_is_refused() {
        let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
            return;
        };
        let buf = ctx.upload_i32_writable(&[0, 0, 0, 0]);
        let error = ctx
            .write_f32(&buf, &[1.0, 2.0, 3.0, 4.0])
            .expect_err("an i32-tagged buffer must not accept an f32 write");
        assert!(matches!(
            error,
            RuntimeError::RepresentationMismatch {
                op: "write_f32",
                ..
            }
        ));
    }

    /// A buffer retagged to the write's storage (mirroring `bind_resident`'s slot-cache-hit retag when a
    /// reused slot's lane flips, card 527 review G1) accepts the write cleanly - the check is not
    /// vacuously always-failing, and a legitimate lane flip is not blocked.
    #[test]
    fn write_into_a_buffer_retagged_to_match_binds_cleanly() {
        let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
            return;
        };
        let buf = ctx
            .upload_i32_writable(&[0, 0, 0, 0])
            .with_storage(BufferStorage::f32());
        ctx.write_f32(&buf, &[1.0, 2.0, 3.0, 4.0])
            .expect("a buffer retagged to f32 must accept an f32 write");
    }
}

/// GP (spec 007): a profiled context records a dispatch with positive host wall, byte counts, and
/// (when the adapter has TIMESTAMP_QUERY) a positive device time. The result matches the unprofiled path.
/// In-crate (not `tests/`) because it drives `Context::timestamps_available`, `pub(crate)` only.
mod profiled_dispatch_tests {
    use crate::Context;

    fn spv_for(body: &poot_kernel_ir::Body, name: &str) -> crate::CompiledKernel {
        let dir = std::env::temp_dir().join("poot-runtime-test");
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, name, poot_codegen::Target::SpirvVulkan);
        poot_codegen::compile(body, poot_codegen::Target::SpirvVulkan, &out)
            .expect("compile spirv");
        let bytes = std::fs::read(&out).unwrap();
        poot_codegen::kernel_handle(body, poot_codegen::Target::SpirvVulkan, bytes)
    }

    #[test]
    fn profiled_dispatch_records_stats() {
        use poot_runtime_common::{CallPurpose, TransferDirection};

        let ctx = match Context::new_with_device_timing(64) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let spv = spv_for(&poot_kernel_ir::fixtures::add_kernel(), "add");
        let a = ctx.upload_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = ctx.upload_f32(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let out = ctx.alloc_f32(5);
        let dispatch = ctx
            .build_cached_dispatch(
                "add",
                "add",
                &spv,
                [64, 1, 1],
                [5, 1, 1],
                &[&a, &b],
                &[5, 5],
                &out,
                5,
            )
            .expect("build cached dispatch");
        ctx.submit_cached(&[&dispatch]).expect("submit");
        assert_eq!(
            ctx.download_f32(&out).unwrap(),
            [11.0, 22.0, 33.0, 44.0, 55.0]
        );

        let counters = ctx.execution_counters();
        let h2d = counters.transfer(CallPurpose::Upload, TransferDirection::HostToDevice);
        let d2h = counters.transfer(CallPurpose::Readback, TransferDirection::DeviceToHost);
        assert!(
            h2d.bytes >= 2 * 5 * 4,
            "the two input buffers were uploaded"
        );
        assert!(d2h.bytes >= 5 * 4, "the output buffer was read back");

        if ctx.timestamps_available() {
            let timing = ctx
                .drain_device_timing()
                .expect("timestamps available but drain_device_timing reported nothing");
            assert_eq!(timing.per_dispatch.len(), 1, "one dispatch recorded");
            assert!(timing.sum.as_nanos() > 0, "positive device time");
            eprintln!("device time: {:?}", timing.sum);
        } else {
            eprintln!("TIMESTAMP_QUERY unavailable; device time skipped (FR-005)");
        }
    }

    /// Card 547a: the Card 552 typed device-timing path's per-batch timestamp
    /// resolve/read buffers (`Context::submit_encoded`'s `query_set`/`ts_guard`, built only when
    /// `TIMESTAMP_QUERY` is available and device timing is on) must be visible to `MemoryCounters`
    /// under `BufferRole::Staging`, not only `alloc_f32`. Skips cleanly when the adapter has no
    /// `TIMESTAMP_QUERY` (`query_set` then always stays `None` and charges nothing, which is
    /// correct, not a gap to test here).
    ///
    /// Mutation (recorded here, applied by hand, never left in the tree; card 547a): in
    /// `Context::submit_encoded` (`crates/poot-runtime/src/context/dispatch.rs`), changed `let guard
    /// = self.tag_alloc(BufferRole::Staging, (bytes * 2) as usize);` to charge a fresh, disconnected
    /// `MemoryCounters::new()` instead. Result: RED - "the typed device-timing path's timestamp
    /// buffers must be charged under BufferRole::Staging: before=0, after=0" (no increase).
    /// Reverted: GREEN.
    #[test]
    fn device_timed_batch_charges_its_timestamp_buffers_as_staging() {
        let ctx = match Context::new_with_device_timing(64) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        if !ctx.timestamps_available() {
            eprintln!("TIMESTAMP_QUERY unavailable; skipping (FR-005)");
            return;
        }
        let spv = spv_for(&poot_kernel_ir::fixtures::add_kernel(), "add_ts_staging");
        let a = ctx.upload_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = ctx.upload_f32(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let out = ctx.alloc_f32(5);
        let dispatch = ctx
            .build_cached_dispatch(
                "add_ts_staging",
                "add_ts_staging",
                &spv,
                [64, 1, 1],
                [5, 1, 1],
                &[&a, &b],
                &[5, 5],
                &out,
                5,
            )
            .expect("build cached dispatch");
        let staging_before: poot_runtime_common::MemoryCounterSnapshot = ctx
            .memory()
            .into_iter()
            .find(|&(role, _)| role == poot_runtime_common::BufferRole::Staging)
            .map(|(_, snap)| snap)
            .unwrap_or_default();
        ctx.submit_cached(&[&dispatch]).expect("submit");
        let staging_after: poot_runtime_common::MemoryCounterSnapshot = ctx
            .memory()
            .into_iter()
            .find(|&(role, _)| role == poot_runtime_common::BufferRole::Staging)
            .map(|(_, snap)| snap)
            .unwrap_or_default();
        assert!(
            staging_after.allocations > staging_before.allocations,
            "the typed device-timing path's timestamp buffers must be charged under \
             BufferRole::Staging: before={}, after={}",
            staging_before.allocations,
            staging_after.allocations
        );
    }
}

/// Card 628: the marked packed-dequant-style `a*b - c*d` kernel
/// (`no_contract_probe_kernel(true)`) dispatched on real RADV/ACO hardware, checked against the unfused
/// scalar reference bit for bit on rows specifically chosen to be FMA-sensitive - a row where fusing the
/// second multiply into the subtract (one rounding, what `NoContraction`'s absence would let ACO do) gives
/// a different f32 than computing them separately (two roundings), so a passing dispatch is real evidence
/// the decoration worked, not a coincidence of rows where fusion would not have mattered anyway.
mod no_contract_tests {
    use crate::{Context, KernelBuffer};

    fn spv_for(body: &poot_kernel_ir::Body, name: &str) -> crate::CompiledKernel {
        let dir = std::env::temp_dir().join("poot-runtime-test");
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, name, poot_codegen::Target::SpirvVulkan);
        poot_codegen::compile(body, poot_codegen::Target::SpirvVulkan, &out)
            .expect("compile spirv");
        let bytes = std::fs::read(&out).unwrap();
        poot_codegen::kernel_handle(body, poot_codegen::Target::SpirvVulkan, bytes)
    }

    /// `n` `(a, b, c, d)` rows where `a*b - c*d` computed with two separate f32 roundings (`m1 = a*b`,
    /// `m2 = c*d`, `r = m1 - m2`) differs from the same expression with the second multiply and the
    /// subtract fused into one rounding step (`(-c).mul_add(d, m1)`, exactly what contracting them into an
    /// FMA computes) - i.e. genuinely FMA-sensitive rows, found by a small deterministic search (splitmix64)
    /// so the test is provably a detector rather than an assumption that some row happens to diverge.
    fn fma_sensitive_rows(n: usize) -> Vec<(f32, f32, f32, f32)> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next_unit = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // [0.5, 2.0): keeps every product well within f32 range with plenty of rounding to bite on.
            0.5 + ((z >> 40) as f32 / (1u32 << 24) as f32) * 1.5
        };
        let mut rows = Vec::with_capacity(n);
        while rows.len() < n {
            let (a, b, c, d) = (next_unit(), next_unit(), next_unit(), next_unit());
            let unfused = (a * b) - (c * d);
            let fused = (-c).mul_add(d, a * b);
            if unfused.to_bits() != fused.to_bits() {
                rows.push((a, b, c, d));
            }
        }
        rows
    }

    #[test]
    fn spirv_no_contract_marker_matches_unfused_reference_on_fma_sensitive_rows() {
        let ctx = match Context::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let rows = fma_sensitive_rows(64);
        let a: Vec<f32> = rows.iter().map(|r| r.0).collect();
        let b: Vec<f32> = rows.iter().map(|r| r.1).collect();
        let c: Vec<f32> = rows.iter().map(|r| r.2).collect();
        let d: Vec<f32> = rows.iter().map(|r| r.3).collect();
        let want: Vec<f32> = rows.iter().map(|&(a, b, c, d)| a * b - c * d).collect();

        let spv = spv_for(
            &poot_test_util::kernel_fixtures::no_contract_probe_kernel(true),
            "no_contract_marked",
        );
        let mut bufs = [
            KernelBuffer::read_only_f32(&a),
            KernelBuffer::read_only_f32(&b),
            KernelBuffer::read_only_f32(&c),
            KernelBuffer::read_only_f32(&d),
            KernelBuffer::write_f32(rows.len()),
        ];
        ctx.dispatch(
            "no_contract_probe",
            &spv,
            [64, 1, 1],
            [rows.len() as u32, 1, 1],
            &mut bufs,
        )
        .expect("dispatch");
        assert_eq!(
            bufs[4].as_f32(),
            &want[..],
            "SC-002: the NoContraction-marked SPIR-V kernel diverged from the unfused scalar reference on \
             an FMA-sensitive row - RADV/ACO contracted the marked subtract with one of its producing \
             multiplies despite the decoration"
        );
    }

    /// The mutation half of SC-002 (dropping the marker) as a real-hardware detector, not just the
    /// structural absence check in `poot-codegen`'s `no_contract.rs`: with `no_contract_probe_kernel(false)`
    /// (the unmarked body), RADV/ACO fuses the second multiply into the subtract by default (confirms
    /// dquant.md 3.3's R1 risk is real on this hardware, not merely theoretical), so the unmarked dispatch
    /// diverges from the unfused reference on the same FMA-sensitive rows the marked test uses.
    #[test]
    fn spirv_unmarked_probe_diverges_from_unfused_reference_on_the_same_rows() {
        let ctx = match Context::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let rows = fma_sensitive_rows(64);
        let a: Vec<f32> = rows.iter().map(|r| r.0).collect();
        let b: Vec<f32> = rows.iter().map(|r| r.1).collect();
        let c: Vec<f32> = rows.iter().map(|r| r.2).collect();
        let d: Vec<f32> = rows.iter().map(|r| r.3).collect();
        let want: Vec<f32> = rows.iter().map(|&(a, b, c, d)| a * b - c * d).collect();

        let spv = spv_for(
            &poot_test_util::kernel_fixtures::no_contract_probe_kernel(false),
            "no_contract_unmarked",
        );
        let mut bufs = [
            KernelBuffer::read_only_f32(&a),
            KernelBuffer::read_only_f32(&b),
            KernelBuffer::read_only_f32(&c),
            KernelBuffer::read_only_f32(&d),
            KernelBuffer::write_f32(rows.len()),
        ];
        ctx.dispatch(
            "no_contract_probe",
            &spv,
            [64, 1, 1],
            [rows.len() as u32, 1, 1],
            &mut bufs,
        )
        .expect("dispatch");
        assert_ne!(
            bufs[4].as_f32(),
            &want[..],
            "expected the unmarked probe to diverge from the unfused reference on at least one \
             FMA-sensitive row (RADV/ACO contracting by default) - if this now matches, either this \
             hardware/driver stopped contracting by default (re-check the marked test still adds value) or \
             `fma_sensitive_rows` needs revisiting"
        );
    }
}

mod require_backend_tests {
    use crate::require_gpu_check_with;

    #[test]
    fn wgpu_context_open_is_required_by_its_own_variable_only() {
        use poot_runtime_common::DeviceBackend;
        let fails_open = |variable: &'static str| {
            std::panic::catch_unwind(|| {
                require_gpu_check_with(|name| (name == variable).then(|| "1".into()), "no device")
            })
            .is_err()
        };
        assert!(fails_open(DeviceBackend::Wgpu.variable()));
        assert!(
            !fails_open("POOT_REQUIRE_GPU"),
            "the retired all-backend switch must not require Wgpu"
        );
        for other in DeviceBackend::ALL
            .into_iter()
            .filter(|other| *other != DeviceBackend::Wgpu)
        {
            assert!(
                !fails_open(other.variable()),
                "{other:?} must not require Wgpu"
            );
        }
    }
}

/// Card 144 / spec 134 P0.75: real dispatch (not just static llc/spirv-val) of the vec4-load SPIR-V shape
/// from `tests/fixtures/dispatch_probe/`, through `Context` on RADV with a readback check. In-crate (not
/// `tests/`) because it drives `KernelBuffer::read_write_f32`, `pub(crate)` only (dead-pub has no
/// exemption for a pub item an integration test alone would keep alive). The fixture is hand-written raw
/// LLVM IR (not a `poot_kernel_ir::Body`), so this shells out to `llc` directly, skip-if-absent, same
/// convention as `poot-codegen/tests/llc.rs`; `tests/dispatch_probe.rs` keeps the other two (negative and
/// mixed-scalar-vec4) cases, which need no `read_write_f32`.
mod vec4_probe_tests {
    use std::path::PathBuf;
    use std::process::Command;

    use crate::{ArgAccess, ArgSchema, CompiledKernel, Context, KernelBuffer};
    use poot_runtime_common::KernelCode;

    fn have(bin: &str) -> bool {
        Command::new(bin).arg("--version").output().is_ok()
    }

    fn probe_dir() -> PathBuf {
        let d = std::env::temp_dir().join("poot-runtime-vec4-probe");
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Compile raw SPIR-V-target LLVM IR text to a `.spv` via system `llc`, then validate it with
    /// `spirv-val`. Panics with the tool's stderr on failure (a hard gate, not a skip; the caller already
    /// checked `have("llc")`).
    fn compile_and_validate(ll_source: &str, tag: &str) -> Vec<u8> {
        let dir = probe_dir();
        let ll_path = dir.join(format!("{tag}.ll"));
        std::fs::write(&ll_path, ll_source).unwrap();
        let spv_path = dir.join(format!("{tag}.spv"));
        let llc = std::env::var("POOT_LLC").unwrap_or_else(|_| "llc".to_string());
        let out = Command::new(&llc)
            .args([
                "-O0",
                "-filetype=obj",
                "--spirv-ext=+SPV_EXT_shader_atomic_float_add",
            ])
            .arg(&ll_path)
            .arg("-o")
            .arg(&spv_path)
            .output()
            .expect("failed to run llc");
        assert!(
            out.status.success(),
            "llc failed for {tag}:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let bytes = std::fs::read(&spv_path).unwrap();
        if have("spirv-val") {
            let v = Command::new("spirv-val")
                .arg("--target-env")
                .arg("vulkan1.3")
                .arg(&spv_path)
                .output()
                .expect("failed to run spirv-val");
            assert!(
                v.status.success(),
                "spirv-val failed for {tag}:\n{}",
                String::from_utf8_lossy(&v.stderr)
            );
        } else {
            eprintln!("spirv-val not on PATH; skipping static validation for {tag}");
        }
        bytes
    }

    /// Wrap hand-written raw-LLVM-IR SPIR-V (not `poot_kernel_ir::Body` + `poot_codegen`) as a
    /// dispatchable [`CompiledKernel`] (card 608's sanctioned "imported kernel" escape hatch: this fixture
    /// is not compiler output, so the test builds the handle itself).
    fn kernel_from_spv(spv: &[u8], args: Vec<ArgSchema>) -> CompiledKernel {
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // SAFETY: `spv` is `llc`/`spirv-val`-clean SPIR-V (checked by `compile_and_validate`) whose
        // bindings match `args` exactly, by construction of this fixture.
        unsafe {
            CompiledKernel::new(
                poot_target::Backend::SpirvVulkan,
                "main",
                KernelCode::SpirvWords(words.into_boxed_slice()),
                args,
                false,
            )
        }
    }

    /// Case 1 (FR-004 / SC-003): a storage buffer declared with a `<4 x float>` element type
    /// (`A1_vec4_typed_buffer_WORKS.ll`, not a bitcast over a scalar-element resource). 64 lanes (one
    /// workgroup) each vec4-load their own 4-float slot, add 1 to every lane, and store it back, checking
    /// the round trip through wgpu/RADV for every lane.
    #[test]
    fn vec4_load_store_runs_on_gpu() {
        if !have("llc") {
            eprintln!("llc not on PATH; skipping");
            return;
        }
        let ctx = match Context::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let ll = include_str!("../tests/fixtures/dispatch_probe/A1_vec4_typed_buffer_WORKS.ll");
        let spv = compile_and_validate(ll, "a1_vec4");

        // 64 lanes * 4 floats/lane = 256 floats; values 1..=256 make every lane's vec4 distinguishable.
        let input: Vec<f32> = (1..=256).map(|i| i as f32).collect();
        let mut bufs = [KernelBuffer::read_write_f32(&input)];
        let kernel = kernel_from_spv(
            &spv,
            vec![ArgSchema::new(
                poot_target::ElementKind::F32,
                ArgAccess::Write,
            )],
        );
        ctx.dispatch("a1_vec4", &kernel, [64, 1, 1], [64, 1, 1], &mut bufs)
            .expect("dispatch");

        let want: Vec<f32> = input.iter().map(|x| x + 1.0).collect();
        assert_eq!(bufs[0].as_f32(), &want[..], "vec4 load+store mismatch");
    }
}

#[path = "tests/cached_pass_packing.rs"]
mod cached_pass_packing;

/// Card 547a SC-003: a BF16 buffer written and read through the storage-typed primitives
/// (`alloc_storage`/`write_bytes`/`read_bytes`) round-trips bit for bit, with exactly `elems * 2`
/// bytes read.
///
/// Mutation (recorded here, applied by hand against real hardware, never left in the tree): in this
/// test, requested `elems * 4` bytes (as if every element were a 4-byte f32 lane) instead of `elems *
/// 2`. Result: RED - wgpu's own validation layer panicked before any native copy ("Copy at offset 0
/// for 68 bytes would end up overrunning the bounds of the Source buffer of size 36"; 17 elements:
/// `36` is the real BF16 allocation, padded from 34 to the next 4-byte multiple, `68 = 17*4` the
/// wrongly-doubled request). Reverted: GREEN, exact bit-for-bit round trip at the true `elems * 2`
/// byte count.
#[test]
fn bf16_round_trips_through_the_storage_typed_primitives_with_exactly_elems_times_2_bytes_gpu() {
    use crate::{BufferStorage, Context};
    use poot_runtime_common::{BufferRole, DeviceBackend};
    use poot_test_util::device_skip::open_or_skip;

    let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
        return;
    };
    let elems = 17usize; // odd count: a 4-byte-lane mutation would desync the tail element.
    let want_bits: Vec<u16> = (0..elems as u16).map(|i| 0x3F00 + i).collect();
    let bytes: Vec<u8> = want_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    assert_eq!(bytes.len(), elems * 2, "fixture bytes must be elems * 2");

    let buf = ctx.alloc_storage(BufferRole::Activation, BufferStorage::bf16(), elems);
    ctx.write_bytes(&buf, &bytes).expect("write_bytes");

    let mut got = vec![0u8; elems * 2];
    ctx.read_bytes(&buf, &mut got).expect("read_bytes");
    assert_eq!(
        got.len(),
        elems * 2,
        "read_bytes must read exactly elems * 2 bytes for a bf16-width buffer"
    );
    assert_eq!(
        got, bytes,
        "bf16 storage-typed round trip must be bit-exact"
    );
    let got_bits: Vec<u16> = got
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(got_bits, want_bits);
}

/// Card 547a: `read_bytes` with an `out` longer than the source buffer's byte capacity must
/// return the typed [`RuntimeError::RangeOutOfBounds`] (mirroring [`Context::write_bytes`]'s own
/// `checked_write_range` check), never panic inside wgpu's own validation layer. A BF16 buffer (odd
/// element count, so its allocated capacity is itself padded to a 4-byte multiple below the
/// requested-but-refused size) is the same shape an earlier bug had: `out.len()` not a multiple of
/// 4 used to make the device-side padded copy overrun a source whose own capacity was never checked.
///
/// Mutation (recorded here, applied by hand against real hardware, never left in the tree; card 547a): in `Context::read_bytes` (`crates/poot-runtime/src/context/readback.rs`), removed the
/// `checked_write_range("read_bytes", b, 0, out.len(), 1)?;` call at the top of the function (the
/// `size`/`padded_capacity` cap stays, so the device-side copy itself does not overrun - it is the
/// final `out.copy_from_slice(&data[..out.len()])` that reads past the now-shorter mapped range).
/// Result: RED - a raw Rust slice-index panic instead of this test's `.expect_err(...)` observing a
/// clean `Err`: `thread '...' panicked at crates/poot-runtime/src/context/readback.rs:190:38: range
/// end index 68 out of range for slice of length 36` (17 bf16 elements: 34 real bytes padded to 36;
/// `68 = 17*4` the oversized request). Reverted: GREEN.
#[test]
fn read_bytes_past_the_buffers_end_is_a_typed_refusal_not_a_wgpu_panic_gpu() {
    use crate::{BufferStorage, Context};
    use poot_runtime_common::{BufferRole, DeviceBackend};
    use poot_test_util::device_skip::open_or_skip;

    let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
        return;
    };
    let elems = 17usize; // odd count: real capacity (34 bytes) is itself not a multiple of 4.
    let buf = ctx.alloc_storage(BufferRole::Activation, BufferStorage::bf16(), elems);

    let mut too_long = vec![0u8; elems * 4]; // real capacity is elems * 2 = 34 bytes.
    let err = ctx
        .read_bytes(&buf, &mut too_long)
        .expect_err("a read past the buffer's byte capacity must be refused, not panic");
    assert!(
        matches!(err, RuntimeError::RangeOutOfBounds { .. }),
        "expected RangeOutOfBounds, got {err:?}"
    );

    // The in-bounds read this refusal guards must still work.
    let mut in_bounds = vec![0u8; elems * 2];
    ctx.read_bytes(&buf, &mut in_bounds)
        .expect("an in-bounds read_bytes must still succeed");
}

/// Card 552 SC-007: the typed, purpose-tagged [`poot_runtime_common::ExecutionCounters`] reconcile
/// against real runtime-call events - an extra readback submit/wait changes `Readback` counters
/// without moving the logical dispatch count or `alloc_f32`'s own allocation count, and a known
/// mixed-purpose submission ([`crate::Context::dispatch`], which reads writable buffers back in the
/// same `queue.submit` as its compute dispatch) counts once in the physical submit total.
mod exec_counters_tests {
    use crate::{Context, KernelBuffer};
    use poot_runtime_common::{CallPurpose, DeviceBackend, TransferDirection};
    use poot_test_util::device_skip::open_or_skip;

    fn spv_for(body: &poot_kernel_ir::Body, name: &str) -> crate::CompiledKernel {
        let dir = std::env::temp_dir().join("poot-runtime-test");
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, name, poot_codegen::Target::SpirvVulkan);
        poot_codegen::compile(body, poot_codegen::Target::SpirvVulkan, &out)
            .expect("compile spirv");
        let bytes = std::fs::read(&out).unwrap();
        poot_codegen::kernel_handle(body, poot_codegen::Target::SpirvVulkan, bytes)
    }

    /// MUTATION (recorded here, not left in the tree; Card 552 SC-007): in
    /// `Context::download_f32` (`context/readback.rs`), change `self.record_submit(CallPurpose::
    /// Readback)` to `self.record_submit(CallPurpose::Compute)` (omit the dedicated Readback
    /// purpose, folding it into compute-only submits as the total). Result: RED - this test's
    /// `assert_eq!(after.calls(CallPurpose::Readback).submits, 1)` fails with `left: 0, right: 1`
    /// (the readback submit is counted under `Compute` instead). Reverted: GREEN.
    #[test]
    fn extra_readback_moves_readback_counters_not_compute_or_allocations_gpu() {
        let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
            return;
        };
        let spv = spv_for(&poot_kernel_ir::fixtures::add_kernel(), "add_exec_counters");
        let a = [1.0f32, 2.0, 3.0];
        let b = [10.0f32, 20.0, 30.0];
        let mut bufs = [
            KernelBuffer::read_only_f32(&a),
            KernelBuffer::read_only_f32(&b),
            KernelBuffer::write_f32(a.len()),
        ];
        ctx.dispatch("add", &spv, [64, 1, 1], [a.len() as u32, 1, 1], &mut bufs)
            .expect("dispatch");

        let before = ctx.execution_counters();
        let out = ctx.alloc_f32(a.len());
        let before_allocs = ctx.buffer_allocs();
        ctx.download_f32(&out).expect("extra readback");
        let after = ctx.execution_counters();

        // The extra readback's own submit/wait/transfer moved under `Readback`...
        assert_eq!(
            after.calls(CallPurpose::Readback).submits
                - before.calls(CallPurpose::Readback).submits,
            1
        );
        assert_eq!(
            after.calls(CallPurpose::Readback).waits - before.calls(CallPurpose::Readback).waits,
            1
        );
        assert_eq!(
            after
                .transfer(CallPurpose::Readback, TransferDirection::DeviceToHost)
                .bytes
                - before
                    .transfer(CallPurpose::Readback, TransferDirection::DeviceToHost)
                    .bytes,
            (a.len() * 4) as u64
        );
        // ...without moving the logical dispatch count (no compute dispatch happened) or the
        // allocation count `alloc_f32` controls (that one `alloc_f32` call above is accounted by
        // `buffer_allocs`, not by this readback).
        assert_eq!(after.logical_dispatches, before.logical_dispatches);
        assert_eq!(ctx.buffer_allocs(), before_allocs, "no new alloc_f32 here");
    }

    /// A known mixed-purpose submission (`Context::dispatch`'s one `queue.submit` also carries the
    /// writable-buffer readback copy) counts once in the physical submit total, never split across
    /// `Compute` and `Readback`.
    ///
    /// MUTATION (recorded here, not left in the tree; Card 552 SC-007): in `Context::dispatch`,
    /// additionally call `self.record_submit(CallPurpose::Readback)` right after the existing
    /// `record_submit(CallPurpose::Compute)` call for the same submission. Result: RED - this test's
    /// `assert_eq!(after.total_submits() - before.total_submits(), 1)` fails with `left: 2, right:
    /// 1` (the one physical submission double-counted). Reverted: GREEN.
    #[test]
    fn one_mixed_purpose_submission_counts_once_in_the_total_gpu() {
        let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
            return;
        };
        let spv = spv_for(
            &poot_kernel_ir::fixtures::add_kernel(),
            "add_exec_counters_mixed",
        );
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let mut bufs = [
            KernelBuffer::read_only_f32(&a),
            KernelBuffer::read_only_f32(&b),
            KernelBuffer::write_f32(a.len()),
        ];
        let before = ctx.execution_counters();
        ctx.dispatch("add", &spv, [64, 1, 1], [a.len() as u32, 1, 1], &mut bufs)
            .expect("dispatch");
        let after = ctx.execution_counters();
        assert_eq!(
            after.total_submits() - before.total_submits(),
            1,
            "one real queue.submit, however many purposes it served"
        );
    }
}

/// Card 552 review F1/F2/F6: one submit path (`Context::submit_encoded`) serves both the
/// counters-only default and the typed detailed-timing path, and `Context::discard_device_timing`
/// prevents a failed step's registrations from leaking into the next one's drain.
mod device_timing_tests {
    use crate::{CachedDispatch, CompiledKernel, Context, DeviceBuffer};
    use poot_runtime_common::DeviceBackend;
    use poot_test_util::device_skip::open_or_skip;

    fn kernel(body: &poot_kernel_ir::Body, name: &str) -> CompiledKernel {
        let dir = std::env::temp_dir().join("poot-runtime-device-timing-test");
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

    /// One two-dispatch batch (add, then square) run on a fresh context, returning the output and
    /// the `submits()` delta `submit_cached` touched (the diagnostic launch-tax counter: one
    /// `queue.submit` per `submit_cached` call).
    fn run_batch(ctx: &Context, seed: f32) -> (Vec<f32>, usize) {
        let add = kernel(&poot_kernel_ir::fixtures::add_kernel(), "dt-add");
        let square = kernel(
            &poot_test_util::kernel_fixtures::square_kernel(),
            "dt-square",
        );
        let a = ctx.upload_f32(&[seed, seed + 1.0]);
        let b = ctx.upload_f32(&[seed + 2.0, seed + 3.0]);
        let sum = ctx.alloc_f32(2);
        let out = ctx.alloc_f32(2);
        let d1 = cached(ctx, "dt-0", "dt-add", &add, &[&a, &b], &sum, 2);
        let d2 = cached(ctx, "dt-1", "dt-square", &square, &[&sum], &out, 2);
        let before = ctx.submits();
        ctx.submit_cached(&[&d1, &d2]).unwrap();
        ctx.poll_wait().unwrap();
        (ctx.download_f32(&out).unwrap(), ctx.submits() - before)
    }

    /// Card 552: the one `submit_encoded` path - a counters-only context and a detailed-
    /// timing context running the identical batch produce bit-identical output and the same
    /// `submits()` delta (one `queue.submit` per `submit_cached` call either way); only the compute-
    /// pass *shape* differs (batched vs per-dispatch), never the submission/output cost, which
    /// must stay unchanged.
    ///
    /// MUTATION (recorded here, not left in the tree; Card 552): in `submit_encoded`,
    /// change `let attributed = self.profiler.is_some() || self.device_timing.get();` to `let
    /// attributed = self.profiler.is_some();` (ignore `device_timing` for the pass-shape/query-set
    /// decision - the bug this review flagged before the fix). Result: RED - the detailed run's
    /// `device_time()` stays `Unknown` because no query set was ever built, so this test's
    /// `assert!(matches!(detailed_time, DeviceTime::Measured(_)))` fails. Reverted: GREEN.
    #[test]
    fn counters_only_and_detailed_contexts_agree_on_output_and_submit_count_gpu() {
        let Some(plain) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
            return;
        };
        let Some(detailed) = open_or_skip(DeviceBackend::Wgpu, Context::new_with_device_timing(4))
        else {
            return;
        };

        let (plain_out, plain_submits) = run_batch(&plain, 1.0);
        let (detailed_out, detailed_submits) = run_batch(&detailed, 1.0);

        assert_eq!(
            plain_out, detailed_out,
            "identical inputs must give identical outputs"
        );
        assert_eq!(
            plain_submits, detailed_submits,
            "one submit_cached call must cost one queue.submit either way"
        );

        // The detailed context actually collected something (proving the query-set/timestamp path
        // really ran, not just that it was skipped identically to the plain path).
        let detailed_time = detailed.drain_device_timing();
        assert!(
            detailed_time.is_some(),
            "a detailed context must have collected per-dispatch device time"
        );
        assert!(
            plain.drain_device_timing().is_none(),
            "a plain context must never collect device timing"
        );
    }

    /// Card 552: a step that registers pending device timing and then fails (here,
    /// simulated by discarding instead of draining - the same call `WgpuDevice::replay`/`abort`
    /// make on a real failure) must not contaminate the next step's drain. Without
    /// `discard_device_timing`, the next step's `drain_device_timing` would sum both steps'
    /// dispatches and misattribute the first step's ticks onto the second step's (different)
    /// dispatch indices.
    ///
    /// MUTATION (recorded here, not left in the tree; Card 552): comment out the
    /// `ctx.discard_device_timing();` call below (matching the pre-fix `WgpuDevice::replay`, which
    /// never discarded). Result: RED - the second batch's drain reports 4 dispatches (2 leaked from
    /// the "failed" first batch + 2 of its own) instead of 2, so this test's
    /// `assert_eq!(second.per_dispatch.len(), 2)` fails with `left: 4, right: 2`. Restored: GREEN.
    #[test]
    fn discard_device_timing_prevents_cross_step_contamination_gpu() {
        let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new_with_device_timing(4))
        else {
            return;
        };

        // Step 1: registers pending device timing, then "fails" (discarded, never drained) - the
        // same thing a real kernel-assert fault or a mid-replay submit error would trigger via
        // `WgpuDevice::replay`'s own leading `discard_device_timing` call on the *next* step, or
        // `abort`'s defensive one on this step.
        let (_out1, ..) = run_batch(&ctx, 10.0);
        ctx.discard_device_timing();

        // Step 2: a fresh batch. Its drain must reflect only its own two dispatches.
        let (_out2, ..) = run_batch(&ctx, 20.0);
        let second = ctx
            .drain_device_timing()
            .expect("step 2 must have collected its own device timing");
        assert_eq!(
            second.per_dispatch.len(),
            2,
            "step 2's drain must not include step 1's discarded dispatches"
        );
        assert_eq!(
            second.per_dispatch.iter().map(|(i, _)| *i).max(),
            Some(1),
            "step 2's dispatch indices must restart at 0, not continue from step 1's"
        );
    }

    /// Card 552: a declared `max_in_flight_queries` bound of 1 is actually enforced - two
    /// `submit_cached` calls before any drain register two pending readbacks, which must not both
    /// stay pending at once; the second registration drains the first early (an extra `waits()`),
    /// and the step's own final `drain_device_timing` still sums both dispatches correctly (no
    /// resource is ever reused while still in flight, no data is lost to the early drain).
    ///
    /// MUTATION (recorded here, not left in the tree; Card 552): in `Context::
    /// drain_for_capacity`, change `while self.pending_device_timing.borrow().len() >= bound` to
    /// `while false` (never enforce the bound - the pre-fix dead-config behavior). Result: RED -
    /// this test's `assert!(waits_after > waits_before)` fails (no early drain ever happened, so
    /// `waits()` does not move between the two `submit_cached` calls). Reverted: GREEN.
    #[test]
    fn max_in_flight_queries_bound_is_enforced_gpu() {
        let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new_with_device_timing(1))
        else {
            return;
        };
        let add = kernel(&poot_kernel_ir::fixtures::add_kernel(), "dt-inflight-add");
        let a = ctx.upload_f32(&[1.0, 2.0]);
        let b = ctx.upload_f32(&[3.0, 4.0]);
        let out1 = ctx.alloc_f32(2);
        let out2 = ctx.alloc_f32(2);
        let d1 = cached(
            &ctx,
            "dt-inflight-0",
            "dt-inflight-add",
            &add,
            &[&a, &b],
            &out1,
            2,
        );
        let d2 = cached(
            &ctx,
            "dt-inflight-1",
            "dt-inflight-add",
            &add,
            &[&a, &b],
            &out2,
            2,
        );

        ctx.submit_cached(&[&d1]).unwrap();
        let waits_before = ctx.execution_counters().total_waits();
        // With the bound at 1, this second registration must drain the first early.
        ctx.submit_cached(&[&d2]).unwrap();
        let waits_after = ctx.execution_counters().total_waits();
        assert!(
            waits_after > waits_before,
            "a second registration past the bound of 1 must poll/drain the first early"
        );

        ctx.poll_wait().unwrap();
        let result = ctx
            .drain_device_timing()
            .expect("both submissions' device timing must still be collected");
        assert_eq!(
            result.per_dispatch.len(),
            2,
            "the early-drained first dispatch and the final-drained second must both be present"
        );
    }
}
