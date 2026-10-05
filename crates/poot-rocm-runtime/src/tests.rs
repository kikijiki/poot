use std::cell::Cell;

use super::*;

#[test]
fn hsa_queue_constants_match_the_public_abi() {
    assert_eq!(bindings::HSA_QUEUE_TYPE_MULTI, 0);
    assert_eq!(bindings::HSA_QUEUE_TYPE_SINGLE, 1);
    assert_eq!(bindings::HSA_QUEUE_TYPE_COOPERATIVE, 2);
    assert_eq!(bindings::HSA_QUEUE_FEATURE_KERNEL_DISPATCH, 1);
}

#[test]
fn queue_role_and_ring_provenance_are_independent_contracts() {
    assert_eq!(
        queue_request(QueueCreationPlan::CooperativeGws, 16_384),
        QueueRequest {
            queue_type: 2,
            size: 16_384,
        }
    );
    assert_eq!(
        queue_request(QueueCreationPlan::RecordedReplay, 16_384),
        QueueRequest {
            queue_type: 0,
            size: 16_384,
        }
    );
    assert_eq!(
        queue_request(QueueCreationPlan::PcieDeviceMemoryRingReceipt, 16_384),
        QueueRequest {
            queue_type: 0,
            size: 16_384,
        }
    );
    assert_eq!(
        QueueCreationPlan::RecordedReplay.provenance(),
        QueueRingProvenance::OrdinarySystemMemory
    );
    assert_eq!(
        QueueCreationPlan::PcieDeviceMemoryRingReceipt.provenance(),
        QueueRingProvenance::PcieDeviceMemoryReceipt
    );
}

#[test]
fn ordinary_ring_provenance_fails_closed_on_any_device_override() {
    assert!(!device_ring_override_present(|_| None));
    assert!(device_ring_override_present(|_| Some("1".into())));
    assert!(device_ring_override_present(|_| Some("0".into())));
    assert!(device_ring_override_present(|_| Some(OsString::new())));

    validate_queue_plan_provenance(QueueCreationPlan::RecordedReplay, false, false).unwrap();
    for (override_at_load, override_now) in [(true, true), (true, false), (false, true)] {
        assert!(matches!(
            validate_queue_plan_provenance(
                QueueCreationPlan::RecordedReplay,
                override_at_load,
                override_now,
            ),
            Err(RocmError::QueueRingProvenanceConflict {
                role: QueueRole::RecordedReplay,
                ..
            })
        ));
    }

    validate_queue_plan_provenance(QueueCreationPlan::PcieDeviceMemoryRingReceipt, true, true)
        .unwrap();
    assert!(
        validate_queue_plan_provenance(
            QueueCreationPlan::PcieDeviceMemoryRingReceipt,
            false,
            true,
        )
        .is_err()
    );
}

#[test]
fn returned_queue_contract_is_validated_before_use() {
    let request = queue_request(QueueCreationPlan::RecordedReplay, 16_384);
    validate_created_queue(
        request,
        QueueProperties {
            queue_type: 0,
            features: 1,
            size: 1_024,
        },
    )
    .expect("a smaller returned ring is valid when its real properties are admitted");

    for actual in [
        QueueProperties {
            queue_type: 2,
            features: 1,
            size: 1_024,
        },
        QueueProperties {
            queue_type: 0,
            features: 0,
            size: 1_024,
        },
        QueueProperties {
            queue_type: 0,
            features: 1,
            size: 0,
        },
        QueueProperties {
            queue_type: 0,
            features: 1,
            size: 1_000,
        },
    ] {
        assert!(matches!(
            validate_created_queue(request, actual),
            Err(RocmError::QueueContract(_))
        ));
    }
}

#[test]
fn one_queue_reference_is_destroyed_once() {
    let mut queue = std::ptr::NonNull::<hsa_queue_t>::dangling().as_ptr();
    let mut destroyed = 0;
    for _ in 0..2 {
        let _ = destroy_queue_reference(&mut queue, |_| destroyed += 1);
    }
    assert_eq!(destroyed, 1);
    assert!(queue.is_null());
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CleanupEvent {
    Free { runtime: usize, ptr: usize },
    Reader { runtime: usize, handle: u64 },
    Executable { runtime: usize, handle: u64 },
    Shutdown { runtime: usize },
}

struct RecordingOwner {
    runtime: usize,
    events: Arc<Mutex<Vec<CleanupEvent>>>,
}

impl ResourceOwner for RecordingOwner {
    fn free_memory(&self, ptr: *mut c_void) {
        self.events.lock().unwrap().push(CleanupEvent::Free {
            runtime: self.runtime,
            ptr: ptr as usize,
        });
    }

    fn destroy_reader(&self, reader: bindings::hsa_code_object_reader_t) {
        self.events.lock().unwrap().push(CleanupEvent::Reader {
            runtime: self.runtime,
            handle: reader.handle,
        });
    }

    fn destroy_executable(&self, executable: bindings::hsa_executable_t) {
        self.events.lock().unwrap().push(CleanupEvent::Executable {
            runtime: self.runtime,
            handle: executable.handle,
        });
    }
}

impl Drop for RecordingOwner {
    fn drop(&mut self) {
        self.events.lock().unwrap().push(CleanupEvent::Shutdown {
            runtime: self.runtime,
        });
    }
}

fn recording_owner(runtime: usize) -> (Arc<dyn ResourceOwner>, Arc<Mutex<Vec<CleanupEvent>>>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let owner: Arc<dyn ResourceOwner> = Arc::new(RecordingOwner {
        runtime,
        events: Arc::clone(&events),
    });
    (owner, events)
}

fn recorded(events: &Arc<Mutex<Vec<CleanupEvent>>>) -> Vec<CleanupEvent> {
    events.lock().unwrap().clone()
}

fn mock_buffer(ptr: usize, owner: Arc<dyn ResourceOwner>) -> RocmBuffer {
    mock_buffer_sharing(ptr, owner, DevicePoison::default())
}

/// A mock allocation that shares `poison` with the runtime that "created" it, charged to its own
/// fresh [`MemoryCounters`] under `BufferRole::Activation` (Card 547a). Every mock here is 4 bytes
/// (one f32), so a test can read "how many of these are live" as `memory.snapshot(Activation).live_bytes / 4`.
fn mock_buffer_sharing(
    ptr: usize,
    owner: Arc<dyn ResourceOwner>,
    poison: DevicePoison,
) -> RocmBuffer {
    mock_buffer_sharing_with(ptr, owner, poison, &MemoryCounters::new())
}

/// [`mock_buffer_sharing`] charged to a caller-supplied, shared [`MemoryCounters`] instead of a fresh
/// one, so a test can read the live total across several mocks.
fn mock_buffer_sharing_with(
    ptr: usize,
    owner: Arc<dyn ResourceOwner>,
    poison: DevicePoison,
    memory: &MemoryCounters,
) -> RocmBuffer {
    RocmBuffer {
        alloc: Arc::new(RocmAlloc {
            ptr,
            pool: Pool::Fine,
            owner,
            guard: std::mem::ManuallyDrop::new(memory.record_alloc(BufferRole::Activation, 4)),
            poison,
        }),
        elem_count: 1,
        byte_capacity: 4,
        storage: BufferStorage::f32(),
    }
}

/// A finished mock module whose executable destruction goes to `owner`, sharing `poison`.
fn mock_module(owner: Arc<dyn ResourceOwner>, poison: DevicePoison, exec: u64) -> HsacoModule {
    let mut construction = ModuleConstruction::new(owner, poison);
    construction.executable = Some(bindings::hsa_executable_t { handle: exec });
    construction.finish(0x1, 16, 0, 0, "kernel.kd".into(), Arc::from([]))
}

fn is_release(event: &CleanupEvent) -> bool {
    matches!(
        event,
        CleanupEvent::Free { .. } | CleanupEvent::Executable { .. }
    )
}

/// Card 602 SC-002 (model-free half): once the shared poison is set, dropping the last handle of a
/// module or allocation releases neither; on a healthy runtime both are released. Letting
/// `HsacoModule`'s drop destroy its executable unconditionally, or `RocmAlloc`'s drop free
/// unconditionally, turns the poisoned case red.
#[test]
fn poisoned_runtime_neither_frees_allocations_nor_destroys_executables() {
    for poisoned in [false, true] {
        let (owner, events) = recording_owner(7);
        let poison = DevicePoison::default();
        let module = mock_module(Arc::clone(&owner), poison.clone(), 0x71);
        let memory = MemoryCounters::new();
        let buffers = [0x700, 0x710]
            .map(|ptr| mock_buffer_sharing_with(ptr, Arc::clone(&owner), poison.clone(), &memory));
        if poisoned {
            poison.poison();
        }

        drop(module);
        drop(buffers);

        let releases: Vec<CleanupEvent> =
            recorded(&events).into_iter().filter(is_release).collect();
        if poisoned {
            assert_eq!(
                releases,
                vec![],
                "a poisoned runtime must not free or destroy what a hung packet may use"
            );
            assert_eq!(
                memory.snapshot(BufferRole::Activation).live_bytes,
                8,
                "both leaked allocations (4 bytes each) stay live"
            );
        } else {
            assert_eq!(
                releases,
                vec![
                    CleanupEvent::Executable {
                        runtime: 7,
                        handle: 0x71,
                    },
                    CleanupEvent::Free {
                        runtime: 7,
                        ptr: 0x700,
                    },
                    CleanupEvent::Free {
                        runtime: 7,
                        ptr: 0x710,
                    },
                ]
            );
            assert_eq!(memory.snapshot(BufferRole::Activation).live_bytes, 0);
        }
    }
}

/// Card 602 SC-003: a poisoned context keeps its queue on the leak list instead of destroying it,
/// since a packet may still be in flight; a healthy one destroys it exactly once. Removing the poison
/// check in `TimeoutLeaks::retire_queue` (the queue release of `impl Drop for RocmContext`) turns the
/// poisoned case red.
#[test]
fn poisoned_context_keeps_its_queue_and_healthy_one_destroys_it() {
    for poisoned in [false, true] {
        let mut leaks = TimeoutLeaks::default();
        if poisoned {
            leaks.poison();
        }
        let original = std::ptr::NonNull::<hsa_queue_t>::dangling().as_ptr();
        let mut queue = original;
        let mut destroyed = Vec::new();

        leaks.retire_queue(&mut queue, |q| destroyed.push(q));
        leaks.retire_queue(&mut queue, |q| destroyed.push(q));

        assert!(queue.is_null(), "the context gives up its queue reference");
        if poisoned {
            assert_eq!(destroyed, vec![], "a poisoned queue must not be destroyed");
            assert_eq!(leaks.retired_queues(), &[original]);
        } else {
            assert_eq!(
                destroyed,
                vec![original],
                "a healthy queue is destroyed once"
            );
            assert!(leaks.retired_queues().is_empty());
        }
    }
}

#[test]
fn ownership_allocation_frees_once_after_clones_and_thread_handoff() {
    let (owner, events) = recording_owner(1);
    let memory = MemoryCounters::new();
    let first =
        mock_buffer_sharing_with(0x1000, Arc::clone(&owner), DevicePoison::default(), &memory);
    let middle = first.clone();
    let last = first.clone();
    assert_eq!(
        memory.snapshot(BufferRole::Activation).live_bytes,
        4,
        "clones are not new allocations"
    );
    drop(owner);

    drop(first);
    drop(middle);
    assert!(
        recorded(&events).is_empty(),
        "a live clone must retain the allocation"
    );
    assert_eq!(memory.snapshot(BufferRole::Activation).live_bytes, 4);

    std::thread::spawn(move || drop(last)).join().unwrap();
    assert_eq!(memory.snapshot(BufferRole::Activation).live_bytes, 0);
    assert_eq!(
        recorded(&events),
        vec![
            CleanupEvent::Free {
                runtime: 1,
                ptr: 0x1000,
            },
            CleanupEvent::Shutdown { runtime: 1 },
        ]
    );
}

#[test]
fn ownership_module_destroys_executable_before_runtime_after_thread_handoff() {
    let (owner, events) = recording_owner(2);
    let mut construction = ModuleConstruction::new(Arc::clone(&owner), DevicePoison::default());
    construction.reader = Some(bindings::hsa_code_object_reader_t { handle: 0x21 });
    construction.executable = Some(bindings::hsa_executable_t { handle: 0x22 });
    let module = construction.finish(0x23, 4, 5, 6, "kernel.kd".into(), Arc::from([]));

    assert_eq!(
        recorded(&events),
        vec![CleanupEvent::Reader {
            runtime: 2,
            handle: 0x21,
        }]
    );
    drop(owner);
    std::thread::spawn(move || drop(module)).join().unwrap();
    assert_eq!(
        recorded(&events),
        vec![
            CleanupEvent::Reader {
                runtime: 2,
                handle: 0x21,
            },
            CleanupEvent::Executable {
                runtime: 2,
                handle: 0x22,
            },
            CleanupEvent::Shutdown { runtime: 2 },
        ]
    );
}

#[test]
fn ownership_multiple_runtime_owners_cleanup_independently() {
    let (owner_a, events_a) = recording_owner(3);
    let (owner_b, events_b) = recording_owner(4);
    let buffer_a = mock_buffer(0x30, Arc::clone(&owner_a));
    let buffer_b = mock_buffer(0x40, Arc::clone(&owner_b));
    drop(owner_a);
    drop(owner_b);

    drop(buffer_a);
    assert_eq!(
        recorded(&events_a),
        vec![
            CleanupEvent::Free {
                runtime: 3,
                ptr: 0x30,
            },
            CleanupEvent::Shutdown { runtime: 3 },
        ]
    );
    assert!(
        recorded(&events_b).is_empty(),
        "dropping one runtime must not affect another"
    );

    drop(buffer_b);
    assert_eq!(
        recorded(&events_b),
        vec![
            CleanupEvent::Free {
                runtime: 4,
                ptr: 0x40,
            },
            CleanupEvent::Shutdown { runtime: 4 },
        ]
    );
}

#[test]
fn ownership_module_constructor_errors_unwind_every_acquired_handle() {
    let cases = [
        (
            Some(0x51),
            None,
            vec![
                CleanupEvent::Reader {
                    runtime: 5,
                    handle: 0x51,
                },
                CleanupEvent::Shutdown { runtime: 5 },
            ],
        ),
        (
            Some(0x61),
            Some(0x62),
            vec![
                CleanupEvent::Executable {
                    runtime: 5,
                    handle: 0x62,
                },
                CleanupEvent::Reader {
                    runtime: 5,
                    handle: 0x61,
                },
                CleanupEvent::Shutdown { runtime: 5 },
            ],
        ),
    ];

    for (reader, executable, expected) in cases {
        let (owner, events) = recording_owner(5);
        let mut construction = ModuleConstruction::new(Arc::clone(&owner), DevicePoison::default());
        construction.reader = reader.map(|handle| bindings::hsa_code_object_reader_t { handle });
        construction.executable = executable.map(|handle| bindings::hsa_executable_t { handle });
        drop(owner);

        drop(construction);
        assert_eq!(recorded(&events), expected);
    }
}

#[test]
fn ownership_resources_are_send() {
    fn assert_send<T: Send>() {}

    assert_send::<RocmBuffer>();
    assert_send::<HsacoModule>();
}

#[test]
fn rocm_context_open_is_required_by_its_own_variable_only() {
    use poot_runtime_common::DeviceBackend;
    let fails_open = |variable: &'static str| {
        std::panic::catch_unwind(|| {
            require_gpu_check_with(|name| (name == variable).then(|| "1".into()), "no device")
        })
        .is_err()
    };
    assert!(fails_open(DeviceBackend::Rocm.variable()));
    assert!(
        !fails_open("POOT_REQUIRE_GPU"),
        "the retired all-backend switch must not require Rocm"
    );
    for other in DeviceBackend::ALL
        .into_iter()
        .filter(|other| *other != DeviceBackend::Rocm)
    {
        assert!(
            !fails_open(other.variable()),
            "{other:?} must not require Rocm"
        );
    }
}

fn test_buffer(elements: usize, storage: BufferStorage) -> RocmBuffer {
    let (elem_count, byte_capacity) =
        checked_layout("test_buffer", elements, storage).expect("valid test buffer");
    let (owner, _) = recording_owner(0);
    let memory = MemoryCounters::new();
    RocmBuffer {
        alloc: Arc::new(RocmAlloc {
            ptr: 0,
            pool: Pool::Fine,
            owner,
            guard: std::mem::ManuallyDrop::new(
                memory.record_alloc(BufferRole::Activation, byte_capacity as u64),
            ),
            poison: DevicePoison::default(),
        }),
        elem_count,
        byte_capacity,
        storage,
    }
}

fn count_copy(
    buffer: &RocmBuffer,
    expected: BufferStorage,
    offset: usize,
    elements: usize,
    whole: bool,
    calls: &Cell<usize>,
) -> Result<(), RocmError> {
    checked_copy(
        buffer,
        "test_copy",
        expected,
        offset,
        elements,
        whole,
        |_| {
            calls.set(calls.get() + 1);
            Ok(())
        },
    )
}

#[test]
fn rejected_oversized_write_makes_zero_copy_calls() {
    let buffer = test_buffer(2, BufferStorage::f32());
    let calls = Cell::new(0);

    let error = count_copy(&buffer, BufferStorage::f32(), 0, 3, true, &calls)
        .expect_err("oversized whole write must fail");

    assert!(matches!(
        error,
        RocmError::SizeMismatch {
            op: "test_copy",
            have: 2,
            got: 3
        }
    ));
    assert_eq!(
        calls.get(),
        0,
        "rejection entered the host/DMA copy closure"
    );
}

#[test]
fn narrow_buffer_rejected_by_f32_operation_before_copy() {
    let buffer = test_buffer(4, BufferStorage::bf16());
    let calls = Cell::new(0);

    let error = count_copy(&buffer, BufferStorage::f32(), 0, 4, true, &calls)
        .expect_err("bf16 storage must not admit an f32 copy");

    let RocmError::RepresentationMismatch {
        op: "test_copy",
        expected,
        actual,
    } = error
    else {
        panic!("expected RepresentationMismatch, got {error:?}");
    };
    assert_eq!(expected, BufferStorage::f32());
    assert_eq!(actual, BufferStorage::bf16());
    assert_eq!(
        calls.get(),
        0,
        "rejection entered the host/DMA copy closure"
    );
}

/// Card 527, mechanism-level (the production bind-path acceptance evidence is
/// `poot-rocm-gpu`'s `bind_bf16_packed_cache_mismatch_is_a_typed_refusal`, card 527):
/// `bf16_packed()` canonicalizes on `BufferElement::I32` (review F2), the one word ROCm's packed upload
/// actually uses (`upload_i32`). A packed-BF16 value and a plain dense I32 value share native element
/// kind; only `checked_copy`'s full `BufferStorage` comparison (not element kind alone) tells them
/// apart.
#[test]
fn bf16_packed_word_rejected_by_dense_i32_bind() {
    let buffer = test_buffer(7, BufferStorage::bf16_packed());
    let calls = Cell::new(0);

    let error = count_copy(&buffer, BufferStorage::i32(), 2, 3, false, &calls)
        .expect_err("packed bf16 lanes must not admit a dense-i32 copy");

    let RocmError::RepresentationMismatch {
        op: "test_copy",
        expected,
        actual,
    } = error
    else {
        panic!("expected RepresentationMismatch, got {error:?}");
    };
    assert_eq!(expected, BufferStorage::i32());
    assert_eq!(actual, BufferStorage::bf16_packed());
    assert_eq!(actual.element(), expected.element(), "same native element");
    assert_eq!(
        calls.get(),
        0,
        "rejection entered the host/DMA copy closure"
    );
}

#[test]
fn overflowing_partial_offset_makes_zero_copy_calls() {
    let buffer = test_buffer(4, BufferStorage::f32());
    let calls = Cell::new(0);

    let error = count_copy(&buffer, BufferStorage::f32(), usize::MAX, 1, false, &calls)
        .expect_err("overflowing element offset must fail");

    assert!(matches!(
        error,
        RocmError::RangeOverflow {
            op: "test_copy",
            element_offset: usize::MAX,
            elements: 1,
        }
    ));
    assert_eq!(
        calls.get(),
        0,
        "rejection entered the host/DMA copy closure"
    );
}

#[test]
fn immutable_metadata_and_valid_copy_share_one_allocation_contract() {
    let buffer = test_buffer(7, BufferStorage::i32());
    let clone = buffer.clone();
    let calls = Cell::new(0);

    count_copy(&clone, BufferStorage::i32(), 2, 3, false, &calls)
        .expect("valid packed-I32 subrange");

    assert_eq!(buffer.elem_count(), 7);
    assert_eq!(clone.byte_capacity(), 28);
    assert_eq!(clone.element(), poot_target::ElementKind::I32);
    assert!(Arc::ptr_eq(&buffer.alloc, &clone.alloc));
    assert_eq!(calls.get(), 1);
}

#[test]
fn allocation_arithmetic_is_checked_before_native_allocation() {
    assert!(matches!(
        checked_layout("allocate_f32", usize::MAX, BufferStorage::f32()),
        Err(RocmError::ElementCountTooLarge { .. } | RocmError::ByteSizeOverflow { .. })
    ));
}

#[test]
fn raw_byte_range_and_kernarg_size_reject_before_boundary_calls() {
    let raw = test_buffer(16, BufferStorage::raw_bytes());
    let calls = Cell::new(0);
    let error = count_copy(&raw, BufferStorage::raw_bytes(), 15, 2, false, &calls)
        .expect_err("out-of-range raw-byte write must fail");
    assert!(matches!(error, RocmError::RangeOutOfBounds { .. }));
    assert_eq!(calls.get(), 0);

    let kernel = KernelHandle::for_test(0, 24, 0, 0, &[]);
    assert!(matches!(
        validate_kernarg_buffer("test_dispatch", &kernel, &raw),
        Err(RocmError::KernargSizeMismatch {
            expected: 24,
            actual: 16,
            ..
        })
    ));
    let typed = test_buffer(24, BufferStorage::f32());
    let Err(RocmError::RepresentationMismatch {
        expected, actual, ..
    }) = validate_kernarg_buffer("test_dispatch", &kernel, &typed)
    else {
        panic!("expected a RepresentationMismatch for a typed buffer bound as kernarg");
    };
    assert_eq!(expected, BufferStorage::raw_bytes());
    assert_eq!(actual, BufferStorage::f32());
}

#[test]
fn native_buffer_contract_roundtrip_rocm() {
    let ctx = match RocmContext::new() {
        Ok(ctx) => ctx,
        Err(RocmError::LibraryNotFound(msg)) => {
            tracing::info!(
                "poot-rocm-runtime: SKIP native_buffer_contract_roundtrip_rocm (no HSA library): {msg}"
            );
            return;
        }
        Err(RocmError::NoGpuAgent) => {
            tracing::info!(
                "poot-rocm-runtime: SKIP native_buffer_contract_roundtrip_rocm (no GPU agent)"
            );
            return;
        }
        Err(error) => {
            panic!("native_buffer_contract_roundtrip_rocm failed to open HSA: {error}")
        }
    };

    let f32_buffer = ctx
        .upload_f32(&[1.0, 2.0, 3.0, 4.0], BufferRole::Activation)
        .expect("upload f32");
    let mut f32_out = [0.0; 4];
    ctx.download_f32(&f32_buffer, &mut f32_out)
        .expect("download f32");
    assert_eq!(f32_out, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(f32_buffer.byte_capacity(), 16);
    assert_eq!(f32_buffer.element(), BufferElement::F32);

    let i32_buffer = ctx
        .upload_i32(&[1, 2, 3, 4], BufferRole::Activation)
        .expect("upload i32");
    let mut i32_out = [0; 4];
    ctx.download_i32(&i32_buffer, &mut i32_out)
        .expect("download i32");
    assert_eq!(i32_out, [1, 2, 3, 4]);
    assert_eq!(i32_buffer.element(), BufferElement::I32);

    let bf16_buffer = ctx
        .upload_bf16_bytes(&[0, 0, 0x80, 0x3f], BufferRole::Activation)
        .expect("upload bf16 bytes");
    assert_eq!(bf16_buffer.elem_count(), 2);
    assert_eq!(bf16_buffer.byte_capacity(), 4);
    assert_eq!(bf16_buffer.element(), BufferElement::Bf16);
    assert!(matches!(
        ctx.update_f32(&bf16_buffer, &[0.0, 1.0]),
        Err(RocmError::RepresentationMismatch { .. })
    ));
}

#[test]
fn fnv1a_known_value() {
    // The empty string is the standard FNV-1a-64 offset basis.
    assert_eq!(fnv1a(""), 0xcbf29ce484222325);
    // "foobar" -> canonical FNV-1a-64 hash.
    assert_eq!(fnv1a("foobar"), 0x85944171f73967e8);
}

#[test]
fn hsa_status_describe_known_codes() {
    assert_eq!(describe(0), "HSA_STATUS_SUCCESS");
    assert_eq!(describe(0x1004), "HSA_STATUS_ERROR_INVALID_AGENT");
    assert_eq!(describe(0x9999), "HSA_STATUS_ERROR_OTHER");
}

#[test]
fn kernel_symbol_matching_is_exact_except_for_amdhsa_descriptor_suffix() {
    assert!(kernel_symbol_matches("resident.kd", "resident"));
    assert!(kernel_symbol_matches("resident", "resident.kd"));
    assert!(kernel_symbol_matches("resident", "resident"));
    assert!(!kernel_symbol_matches("resident_suffix.kd", "resident"));
    assert!(!kernel_symbol_matches("other.kd", "resident"));
}

// Card 125 (spec 120 FR-004/SC-004): `upload_strategy_for_pool` is a pure function of (location,
// flags), so it is unit-checkable with no HSA context. The DMA transfer itself and the second-target
// run are the hardware gate (docs/tasks/*/125-amd-discrete-upload-dma.md).

#[test]
fn upload_strategy_cpu_pool_is_host_write() {
    // A CPU-located pool is always host-visible regardless of its coherency flags.
    assert_eq!(
        upload_strategy_for_pool(HSA_AMD_MEMORY_POOL_LOCATION_CPU, 0),
        UploadStrategy::HostWrite
    );
    assert_eq!(
        upload_strategy_for_pool(
            HSA_AMD_MEMORY_POOL_LOCATION_CPU,
            HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED
        ),
        UploadStrategy::HostWrite
    );
}

#[test]
fn upload_strategy_gpu_extended_scope_fine_is_host_write() {
    // Strix Halo's aliased unified-memory view: GPU-located, EXTENDED_SCOPE_FINE_GRAINED.
    assert_eq!(
        upload_strategy_for_pool(
            HSA_AMD_MEMORY_POOL_LOCATION_GPU,
            HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED
        ),
        UploadStrategy::HostWrite
    );
    // KERNARG_INIT alongside EXTENDED_SCOPE_FINE_GRAINED (the realistic combination) still counts.
    assert_eq!(
        upload_strategy_for_pool(
            HSA_AMD_MEMORY_POOL_LOCATION_GPU,
            HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED
                | HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
        ),
        UploadStrategy::HostWrite
    );
}

#[test]
fn upload_strategy_gpu_coarse_grained_is_dma() {
    // A discrete GPU's plain VRAM pool: GPU-located, COARSE_GRAINED only.
    assert_eq!(
        upload_strategy_for_pool(
            HSA_AMD_MEMORY_POOL_LOCATION_GPU,
            HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED
        ),
        UploadStrategy::Dma
    );
}

#[test]
fn upload_strategy_gpu_plain_fine_grained_is_dma() {
    // FR-004: a GPU-located pool that is FINE_GRAINED (device-coherent) but not
    // EXTENDED_SCOPE_FINE_GRAINED guarantees device coherency, not host-mappability (e.g. a
    // coherent-fabric multi-GPU VRAM pool on CDNA without an APU-style host-mapped alias). It must
    // resolve to Dma.
    assert_eq!(
        upload_strategy_for_pool(
            HSA_AMD_MEMORY_POOL_LOCATION_GPU,
            HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED
        ),
        UploadStrategy::Dma
    );
}

/// M0 acceptance: build a context, assert the agent picker found a GPU and the ISA name is
/// non-empty. Skips cleanly if ROCm is not installed (no `/dev/kfd`, no `libhsa-runtime64.so.1`, no
/// GPU agent) per FR-004 / FR-005.
#[test]
fn agent_enumeration_works() {
    match RocmContext::new() {
        Ok(ctx) => {
            assert!(
                ctx.gpu_agent.handle != 0,
                "context picked a GPU agent with a zero handle"
            );
            assert!(
                !ctx.isa_name.is_empty(),
                "ISA name was empty; env var HSA_OVERRIDE_GFX_VERSION unset?"
            );
            tracing::info!(
                "poot-rocm-runtime: agent_enumeration_works OK on ISA={}",
                ctx.isa_name
            );
        }
        Err(RocmError::LibraryNotFound(msg)) => {
            tracing::info!(
                "poot-rocm-runtime: SKIP agent_enumeration_works (no HSA library): {msg}"
            );
        }
        Err(RocmError::NoGpuAgent) => {
            tracing::info!("poot-rocm-runtime: SKIP agent_enumeration_works (no GPU agent)");
        }
        Err(e) => panic!("agent_enumeration_works failed: {e}"),
    }
}

/// Card 030: `vram_used_bytes`/`vram_budget_bytes` return plausible values on real gfx1151. Mirrors
/// the wgpu receipt (`vram_used_bytes_reports_a_plausible_value`, `crates/poot-gpu/tests/graph.rs`)
/// and the PTX receipt: budget >= used, both within a device-scale range for this box's ~96 GB-class
/// unified-memory budget (generous bounds; the carveout varies by BIOS/GTT config). Skips cleanly if
/// ROCm is not installed.
#[test]
fn vram_used_bytes_reports_a_plausible_value() {
    let ctx = match RocmContext::new() {
        Ok(ctx) => ctx,
        Err(RocmError::LibraryNotFound(msg)) => {
            tracing::info!(
                "poot-rocm-runtime: SKIP vram_used_bytes_reports_a_plausible_value (no HSA \
                 library): {msg}"
            );
            return;
        }
        Err(RocmError::NoGpuAgent) => {
            tracing::info!(
                "poot-rocm-runtime: SKIP vram_used_bytes_reports_a_plausible_value (no GPU agent)"
            );
            return;
        }
        Err(e) => panic!("vram_used_bytes_reports_a_plausible_value failed: {e}"),
    };

    let budget = ctx
        .vram_budget_bytes()
        .expect("HSA_AMD_MEMORY_POOL_INFO_SIZE sum should return a value on this device");
    eprintln!(
        "ROCm device VRAM budget (pool-size total): {:.1} MiB",
        budget as f64 / (1024.0 * 1024.0)
    );
    assert!(
        budget > 256 * 1024 * 1024 && budget < 256u64 * 1024 * 1024 * 1024,
        "device VRAM budget implausible: {budget} bytes"
    );

    let used = ctx
        .vram_used_bytes()
        .expect("HSA_AMD_AGENT_INFO_MEMORY_AVAIL query should return a value on this device");
    eprintln!(
        "ROCm device VRAM used (budget - MEMORY_AVAIL): {:.1} MiB",
        used as f64 / (1024.0 * 1024.0)
    );
    assert!(used <= budget, "VRAM used {used} > budget {budget}");

    // Allocate a 64 MiB buffer and confirm `used` moves plausibly. On an APU the fine-grained system
    // pool allocated from here is the same pool `HSA_AMD_AGENT_INFO_MEMORY_AVAIL` reports over. Other
    // system activity also moves it, so this only checks the query is live, not an exact delta.
    let before = used;
    let elems = 16 * 1024 * 1024; // 16M f32 = 64 MiB
    let _buf = ctx
        .allocate_f32(elems, BufferRole::Activation)
        .expect("allocate_f32(64 MiB) should succeed");
    let after = ctx
        .vram_used_bytes()
        .expect("vram_used_bytes should still succeed after an allocation");
    eprintln!(
        "ROCm device VRAM used after a 64 MiB allocation: {:.1} MiB (was {:.1} MiB)",
        after as f64 / (1024.0 * 1024.0),
        before as f64 / (1024.0 * 1024.0)
    );
    // Not a strict >: a 96 GB-class shared pool absorbs 64 MiB within noise from co-resident
    // activity, but the value must stay in the same plausible envelope.
    assert!(
        after <= budget,
        "VRAM used {after} > budget {budget} after allocation"
    );
}

/// Open a context for a device test, or `None` (a logged skip) when ROCm or a GPU is absent.
fn open_context_or_skip(test: &str) -> Option<RocmContext> {
    match RocmContext::new() {
        Ok(ctx) => Some(ctx),
        Err(RocmError::LibraryNotFound(msg)) => {
            tracing::info!("poot-rocm-runtime: SKIP {test} (no HSA library): {msg}");
            None
        }
        Err(RocmError::NoGpuAgent) => {
            tracing::info!("poot-rocm-runtime: SKIP {test} (no GPU agent)");
            None
        }
        Err(e) => panic!("{test} failed: {e}"),
    }
}

/// [`open_context_or_skip`] with a 1-second [`RocmContextOptions::wait_timeout`] (Card 548:
/// typed construction-time configuration, never an environment read) instead of the 600s
/// default, so a forced-timeout test runs in ~1-2s.
fn open_context_with_one_second_wait_bound_or_skip(test: &str) -> Option<RocmContext> {
    match RocmContext::new_with_options(RocmContextOptions {
        wait_timeout: std::time::Duration::from_secs(1),
    }) {
        Ok(ctx) => Some(ctx),
        Err(RocmError::LibraryNotFound(msg)) => {
            tracing::info!("poot-rocm-runtime: SKIP {test} (no HSA library): {msg}");
            None
        }
        Err(RocmError::NoGpuAgent) => {
            tracing::info!("poot-rocm-runtime: SKIP {test} (no GPU agent)");
            None
        }
        Err(e) => panic!("{test} failed: {e}"),
    }
}

fn create_signal(ctx: &RocmContext, initial: i64) -> bindings::hsa_signal_t {
    let mut signal = bindings::hsa_signal_t { handle: 0 };
    // SAFETY: zero consumers with a null list is allowed; `signal` is a valid out-pointer.
    let status =
        unsafe { (ctx.hsa.funcs.hsa_signal_create)(initial, 0, std::ptr::null(), &mut signal) };
    check(status).expect("hsa_signal_create failed");
    signal
}

/// The queue's current write index, read without changing it.
fn write_index(ctx: &RocmContext) -> u64 {
    // SAFETY: the context's live queue; adding 0 only reads the index.
    unsafe { (ctx.hsa.funcs.hsa_queue_add_write_index_relaxed)(ctx.queue, 0) }
}

/// A kernel handle and matching kernarg buffer for calls that must be refused or fail before any
/// packet is written. `kernel_object` 0 is never executed by these tests.
fn unexecuted_dispatch(ctx: &RocmContext) -> (KernelHandle, RocmBuffer) {
    let kernel = KernelHandle::for_test(0, 64, 0, 0, &[]);
    let kernarg = ctx.allocate_kernarg(64).expect("allocate_kernarg");
    (kernel, kernarg)
}

/// A poisoned context refuses a submission with the typed error and leaves the queue untouched.
fn assert_submissions_refused_without_touching_the_queue(ctx: &RocmContext, timed_out_path: &str) {
    assert!(
        ctx.is_poisoned(),
        "a timed-out {timed_out_path} wait must poison the context"
    );
    let (kernel, kernarg) = unexecuted_dispatch(ctx);
    let write_index_before = write_index(ctx);
    let dispatch = ctx.dispatch(&kernel, &kernarg, &[], (1, 1, 1), (1, 1, 1));
    assert!(
        matches!(dispatch, Err(RocmError::ContextPoisoned { op: "dispatch" })),
        "dispatch after a {timed_out_path} timeout must return ContextPoisoned, got {dispatch:?}"
    );
    let replay = ctx.replay_graph_batched(&[(kernel, kernarg, (1, 1, 1), (1, 1, 1))], &[&[]]);
    assert!(
        matches!(
            replay,
            Err(RocmError::ContextPoisoned {
                op: "replay_graph_batched"
            })
        ),
        "replay after a {timed_out_path} timeout must return ContextPoisoned, got {replay:?}"
    );
    let synchronize = ctx.synchronize();
    assert!(
        matches!(
            synchronize,
            Err(RocmError::ContextPoisoned { op: "synchronize" })
        ),
        "synchronize after a {timed_out_path} timeout must return ContextPoisoned, got {synchronize:?}"
    );
    assert_eq!(
        write_index(ctx),
        write_index_before,
        "a refused submission must not reserve a queue slot"
    );
}

/// Run a bounded wait on a completion signal that is never decremented, so it must time out.
fn force_completion_wait_timeout(ctx: &RocmContext, held: Held) -> RocmError {
    let completion = create_signal(ctx, 1);
    ctx.wait_completion_bounded(completion, "test", held)
        .expect_err("a signal that is never decremented must time out")
}

/// Regression test for `wait_completion_bounded`: a completion signal that never decrements must
/// surface as `QueueWaitTimeout`, not spin forever, and must not free what the packet may still use:
/// the signal and the staging buffer move to the context's leak list, the context is poisoned, and
/// the kernarg allocation stays live after its last handle drops. The 1-second
/// [`RocmContextOptions::wait_timeout`] keeps it to ~1-2s instead of the 600s default. Skips
/// cleanly if ROCm is not installed. Restoring
/// free-on-timeout (destroy the signal and drop `held`, no leak, no poison) turns it red.
#[test]
fn wait_completion_bounded_times_out() {
    let Some(ctx) =
        open_context_with_one_second_wait_bound_or_skip("wait_completion_bounded_times_out")
    else {
        return;
    };

    let completion = create_signal(&ctx, 1);
    let memory = ctx.hsa.memory.clone();
    let kernarg = ctx.allocate_kernarg(64).expect("allocate_kernarg");
    let live_with_kernarg = memory.total_live_bytes();
    let staging = ctx.dma_staging(4096).expect("dma_staging");
    let staging_ptr = staging.agent_ptr();

    let start = std::time::Instant::now();
    let result = ctx.wait_completion_bounded(
        completion,
        "test",
        Held {
            staging: Some(staging),
        },
    );
    let elapsed = start.elapsed();
    // The caller drops its own handle, as `dispatch`'s caller does when it propagates the error.
    drop(kernarg);

    match result {
        Err(RocmError::QueueWaitTimeout(timeout, tag)) => {
            assert_eq!(tag, "test");
            assert_eq!(timeout, 1);
            tracing::info!(
                "poot-rocm-runtime: wait_completion_bounded_times_out OK, timed out after {:?}",
                elapsed
            );
        }
        other => {
            panic!("expected Err(QueueWaitTimeout(1, \"test\")), got {other:?} after {elapsed:?}")
        }
    }
    assert!(
        ctx.is_poisoned(),
        "a timed-out wait must poison the context"
    );
    let abandoned = ctx.timeouts.abandoned();
    assert_eq!(
        abandoned.len(),
        1,
        "the timed-out submission must be on the leak list"
    );
    let entry = &abandoned[0];
    assert_eq!(
        entry.signal, completion,
        "the leaked signal is the waited one"
    );
    assert_eq!(
        entry.held.staging.as_ref().map(StagingBuffer::agent_ptr),
        Some(staging_ptr),
        "the pinned staging buffer must be on the leak list, not unlocked or freed"
    );
    assert_eq!(
        memory.total_live_bytes(),
        live_with_kernarg,
        "the kernarg allocation must stay alive after the caller dropped its handle"
    );
    // SAFETY: the leaked signal is never destroyed, so it is still live.
    let signal_value = unsafe { (ctx.hsa.funcs.hsa_signal_load_relaxed)(completion) };
    assert_eq!(
        signal_value, 1,
        "the leaked signal must still be a live, undecremented signal"
    );
}

/// After a timed-out completion wait, every later submission returns the typed poisoned error
/// before touching the queue. Skipping the poison in the wait's timeout branch turns it red.
#[test]
fn submissions_after_a_timed_out_completion_wait_are_refused() {
    let Some(ctx) = open_context_with_one_second_wait_bound_or_skip(
        "submissions_after_a_timed_out_completion_wait_are_refused",
    ) else {
        return;
    };

    let error = force_completion_wait_timeout(&ctx, Held::default());

    assert!(matches!(error, RocmError::QueueWaitTimeout(1, "test")));
    assert_submissions_refused_without_touching_the_queue(&ctx, "completion");
}

/// A `synchronize` that outlives its bound poisons the context: the queued packets may still run.
/// Skipping the poison in `synchronize`'s timeout branch turns it red.
#[test]
fn submissions_after_a_timed_out_synchronize_are_refused() {
    let Some(ctx) = open_context_with_one_second_wait_bound_or_skip(
        "submissions_after_a_timed_out_synchronize_are_refused",
    ) else {
        return;
    };
    // Claim a slot that no packet ever fills, so the read index never reaches the write index.
    // SAFETY: the context's live queue; the claimed slot keeps its INVALID header, so the packet
    // processor never consumes it.
    unsafe {
        (ctx.hsa.funcs.hsa_queue_add_write_index_relaxed)(ctx.queue, 1);
    }

    let error = ctx.synchronize().expect_err("synchronize must time out");

    assert!(
        matches!(error, RocmError::QueueWaitTimeout(1, "synchronize")),
        "got {error:?}"
    );
    assert_submissions_refused_without_touching_the_queue(&ctx, "synchronize");
}

/// The queue-space wait of the packet publisher reads the env-overridable bound, and its timeout
/// leaks the completion signal and poisons the context: earlier chunks of the batch may already be
/// in flight. Skipping the poison on this path turns it red.
#[test]
fn submissions_after_a_timed_out_back_pressure_wait_are_refused() {
    let Some(ctx) = open_context_with_one_second_wait_bound_or_skip(
        "submissions_after_a_timed_out_back_pressure_wait_are_refused",
    ) else {
        return;
    };
    let (kernel, kernarg) = unexecuted_dispatch(&ctx);
    claim_every_queue_slot(&ctx);

    let error = ctx
        .replay_graph_batched(&[(kernel, kernarg.clone(), (1, 1, 1), (1, 1, 1))], &[&[]])
        .expect_err("the publisher's queue-space wait must time out");

    assert!(
        matches!(
            error,
            RocmError::QueueWaitTimeout(1, "upload back-pressure")
        ),
        "got {error:?}"
    );
    assert_eq!(ctx.timeouts.abandoned().len(), 1);
    assert_submissions_refused_without_touching_the_queue(&ctx, "back-pressure");
}

/// Claim every slot of the context's queue without publishing a packet: the read index stays at 0,
/// so the next submission's queue-space wait never succeeds and times out under
/// [`open_context_with_one_second_wait_bound_or_skip`].
fn claim_every_queue_slot(ctx: &RocmContext) {
    let queue_size = ctx.queue_layout().size as u64;
    // SAFETY: the context's live queue; the claimed slots keep INVALID headers, so the packet processor
    // never consumes them.
    unsafe {
        (ctx.hsa.funcs.hsa_queue_add_write_index_relaxed)(ctx.queue, queue_size);
    }
}

/// Card 602 SC-002: after a forced dispatch timeout, the kernel's code object and its argument
/// allocations (the kernarg and the data buffers it points at) are not released when their last
/// handles drop. The code object and one data buffer report to a counting `ResourceOwner` double
/// sharing the context's poison, which must record zero `destroy_executable` and zero frees; the
/// real kernarg and data allocation must stay live on the runtime's allocation counter. Letting
/// `HsacoModule`'s drop destroy its executable unconditionally turns it red (one executable event).
#[test]
fn dispatch_timeout_keeps_code_object_and_argument_allocations() {
    let Some(ctx) = open_context_with_one_second_wait_bound_or_skip(
        "dispatch_timeout_keeps_code_object_and_argument_allocations",
    ) else {
        return;
    };
    let (owner, events) = recording_owner(602);
    let module = mock_module(Arc::clone(&owner), ctx.timeouts.device_poison(), 0x602);
    let kernel = ctx.lookup_kernel(&module, "kernel").expect("lookup_kernel");
    let recorded_data =
        mock_buffer_sharing(0x6020, Arc::clone(&owner), ctx.timeouts.device_poison());
    let memory = ctx.hsa.memory.clone();
    let live_before = memory.total_live_bytes();
    let data = ctx
        .allocate_f32(64, BufferRole::Activation)
        .expect("allocate_f32");
    let kernarg = ctx
        .allocate_kernarg(kernel.kernarg_size() as usize)
        .expect("allocate_kernarg");
    let mut args = Vec::new();
    for buffer in [&data, &recorded_data] {
        args.extend_from_slice(&buffer.device_ptr().to_le_bytes());
    }
    ctx.write_raw_bytes(&kernarg, 0, &args)
        .expect("write kernarg");
    let live_with_arguments = memory.total_live_bytes();
    assert_eq!(
        live_with_arguments,
        live_before + data.byte_capacity() as u64 + kernarg.byte_capacity() as u64,
        "exactly the new data and kernarg allocations must be live"
    );
    claim_every_queue_slot(&ctx);

    let error = ctx
        .dispatch(&kernel, &kernarg, &[], (1, 1, 1), (1, 1, 1))
        .expect_err("the dispatch must time out");
    assert!(
        matches!(error, RocmError::QueueWaitTimeout(1, _)),
        "got {error:?}"
    );
    // The executor propagates the error and drops its handles, as on any failed submission.
    drop(module);
    drop(kernarg);
    drop(data);
    drop(recorded_data);

    let releases: Vec<CleanupEvent> = recorded(&events).into_iter().filter(is_release).collect();
    assert_eq!(
        releases,
        vec![],
        "a timed-out dispatch's code object and argument buffers must not be released"
    );
    assert_eq!(
        memory.total_live_bytes(),
        live_with_arguments,
        "the kernarg and data allocations must stay live after their last handles drop"
    );
}

/// Card 602 SC-001: a DMA download whose copy is forced to time out (it waits on a gate signal that
/// is never decremented while the host waits) leaves the caller's buffer to the host. The staging
/// buffer is on the leak list; the caller then reuses its buffer, the test opens the gate so the
/// leaked copy really runs, and every caller byte must still hold what the host wrote while every
/// staging byte holds the device data. The caller's `Vec` stays allocated only so a regression shows
/// as a wrong value rather than heap corruption. DMA-ing straight into the caller's slice again
/// turns it red (the reused bytes are overwritten and the staging buffer holds no device data).
#[test]
fn timed_out_dma_download_leaves_the_caller_buffer_to_the_host() {
    let Some(ctx) = open_context_with_one_second_wait_bound_or_skip(
        "timed_out_dma_download_leaves_the_caller_buffer_to_the_host",
    ) else {
        return;
    };
    let device_bytes: Vec<u8> = (0..4096u32).map(|i| (i * 7 + 3) as u8).collect();
    let source = ctx.allocate_kernarg(device_bytes.len()).expect("allocate");
    ctx.write_raw_bytes(&source, 0, &device_bytes)
        .expect("fill source");
    let gate = create_signal(&ctx, 1);
    let mut caller = vec![0u8; device_bytes.len()];

    // SAFETY: `source` is a live allocation of `caller.len()` bytes and `caller` is writable for as
    // many.
    let error = unsafe {
        ctx.dma_download(
            source.alloc.as_ptr(),
            caller.as_mut_ptr(),
            caller.len(),
            &[gate],
        )
    }
    .expect_err("a copy gated on a never-decremented signal must time out");
    assert!(
        matches!(error, RocmError::QueueWaitTimeout(1, "dma_copy")),
        "got {error:?}"
    );
    // The caller reuses its buffer for other data.
    caller.fill(0x5a);

    let (completion, staging_host, staging_len) = {
        let abandoned = ctx.timeouts.abandoned();
        assert_eq!(abandoned.len(), 1);
        let staging = abandoned[0]
            .held
            .staging
            .as_ref()
            .expect("the timed-out copy's staging buffer must be on the leak list");
        (abandoned[0].signal, staging.host_ptr(), staging.len())
    };
    assert_eq!(staging_len, device_bytes.len());

    // Release the leaked copy and wait for it to land.
    // SAFETY: `gate` is a live signal this test created.
    unsafe {
        (ctx.hsa.funcs.hsa_signal_store_release)(gate, 0);
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    // SAFETY: the leaked completion signal is never destroyed.
    while unsafe { (ctx.hsa.funcs.hsa_signal_load_relaxed)(completion) } >= 1 {
        assert!(
            std::time::Instant::now() < deadline,
            "the released copy must complete"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    std::sync::atomic::fence(Ordering::Acquire);

    let wrong_caller = caller.iter().filter(|&&b| b != 0x5a).count();
    assert_eq!(
        wrong_caller, 0,
        "{wrong_caller} caller bytes were written by the device after the timeout"
    );
    // SAFETY: the leaked staging buffer is never freed, holds `staging_len` initialized bytes, and the
    // copy into it has completed (acquire fence above).
    let staged = unsafe { std::slice::from_raw_parts(staging_host, staging_len) };
    assert_eq!(
        first_diff(staged, &device_bytes),
        None,
        "the leaked copy must land in the staging buffer"
    );
}

/// Card 602: the staged DMA path round-trips every byte. On this APU production uses host writes, so
/// this is the only device run of `dma_upload`/`dma_download`; the upload is also checked through the
/// host-visible allocation, independently of the download.
#[test]
fn staged_dma_transfers_round_trip_every_byte() {
    let Some(ctx) = open_context_or_skip("staged_dma_transfers_round_trip_every_byte") else {
        return;
    };
    let bytes: Vec<u8> = (0..65_537u32).map(|i| (i * 31 + i / 256) as u8).collect();
    let buffer = ctx.allocate_kernarg(bytes.len()).expect("allocate");

    // SAFETY: `buffer` is a live allocation of `bytes.len()` bytes.
    unsafe { ctx.dma_upload(bytes.as_ptr(), buffer.alloc.as_ptr(), bytes.len()) }
        .expect("dma_upload");
    if ctx.upload_strategy == UploadStrategy::HostWrite {
        // SAFETY: under `HostWrite` the allocation is host-visible, `bytes.len()` bytes long, and fully
        // written by the completed upload.
        let resident =
            unsafe { std::slice::from_raw_parts(buffer.alloc.as_ptr() as *const u8, bytes.len()) };
        assert_eq!(first_diff(resident, &bytes), None, "uploaded bytes differ");
    }
    let mut back = vec![0u8; bytes.len()];
    // SAFETY: `buffer` holds `back.len()` bytes and `back` is writable for as many.
    unsafe { ctx.dma_download(buffer.alloc.as_ptr(), back.as_mut_ptr(), back.len(), &[]) }
        .expect("dma_download");
    assert_eq!(first_diff(&back, &bytes), None, "downloaded bytes differ");
    assert!(!ctx.is_poisoned());
}

/// A completion signal that drops while the host is waiting must wake the wait promptly. On gfx1151
/// `hsa_signal_wait_scacquire` sleeps the full `timeout_hint` before re-reading the signal (measured
/// ~1.05s with a 1s hint, for both `BLOCKED` and `ACTIVE`), so a 1s hint made every ROCm dispatch
/// cost ~1s (olmo2 decode-curve TPOT ~2228ms = replay wait + argmax wait). Mutating the wait's hint
/// back to `1_000_000_000` makes this assert fail (elapsed ~1.05s, not the 50ms drop).
#[test]
fn wait_completion_bounded_wakes_when_signal_drops() {
    let ctx = match RocmContext::new() {
        Ok(ctx) => ctx,
        Err(RocmError::LibraryNotFound(msg)) => {
            tracing::info!(
                "poot-rocm-runtime: SKIP wait_completion_bounded_wakes_when_signal_drops (no HSA library): {msg}"
            );
            return;
        }
        Err(RocmError::NoGpuAgent) => {
            tracing::info!(
                "poot-rocm-runtime: SKIP wait_completion_bounded_wakes_when_signal_drops (no GPU agent)"
            );
            return;
        }
        Err(e) => panic!("wait_completion_bounded_wakes_when_signal_drops failed: {e}"),
    };

    let mut completion: bindings::hsa_signal_t = bindings::hsa_signal_t { handle: 0 };
    // SAFETY: zero consumers with a null list is allowed; `completion` is a valid out-pointer.
    let create_status =
        unsafe { (ctx.hsa.funcs.hsa_signal_create)(1, 0, std::ptr::null(), &mut completion) };
    check(create_status).expect("hsa_signal_create failed");

    let dropper = completion;
    // Copy the fn pointer so the thread does not borrow `ctx` (which must outlive it for Drop).
    let store_release: unsafe extern "C" fn(bindings::hsa_signal_t, i64) =
        ctx.hsa.funcs.hsa_signal_store_release;
    let drop_after = std::time::Duration::from_millis(50);
    let dropper_thread = std::thread::spawn(move || {
        std::thread::sleep(drop_after);
        // SAFETY: the signal stays live until the wait destroys it after this store completes it.
        unsafe {
            store_release(dropper, 0);
        }
    });

    let start = std::time::Instant::now();
    let result = ctx.wait_completion_bounded(completion, "test", Held::default());
    let elapsed = start.elapsed();
    dropper_thread.join().expect("dropper thread");

    match result {
        Ok(_) => {}
        other => {
            panic!("expected Ok(_) when the signal drops after {drop_after:?}, got {other:?}")
        }
    }
    assert!(
        elapsed < std::time::Duration::from_millis(500),
        "wait returned after {elapsed:?}; expected to wake when the signal dropped (~{drop_after:?}), \
         not after a full 1s timeout_hint (blocked-wait regression)"
    );
}

/// Card 252 round 8: logs this box's `HSA_AGENT_INFO_QUEUE_MAX_SIZE` and whether
/// `replay_graph_batched`'s `chunk_size = queue_size/2` exceeds granitemoe-tiny's 209-AQL-per-step
/// batched decode graph. If `queue_size/2 > 209`, a decode step never spans more than one chunk here
/// (round 8 addendum).
#[test]
fn replay_graph_batched_chunk_size_vs_granitemoe_step_aql_count() {
    let ctx = match RocmContext::new() {
        Ok(ctx) => ctx,
        Err(RocmError::LibraryNotFound(msg)) => {
            tracing::info!(
                "poot-rocm-runtime: SKIP replay_graph_batched_chunk_size_vs_granitemoe_step_aql_count (no HSA library): {msg}"
            );
            return;
        }
        Err(RocmError::NoGpuAgent) => {
            tracing::info!(
                "poot-rocm-runtime: SKIP replay_graph_batched_chunk_size_vs_granitemoe_step_aql_count (no GPU agent)"
            );
            return;
        }
        Err(e) => {
            panic!("replay_graph_batched_chunk_size_vs_granitemoe_step_aql_count failed: {e}")
        }
    };

    // SAFETY: same cast `replay_graph_batched` uses to read `q.size`, per the HSA spec v1.2 layout
    // `QueueLayout` documents.
    let q = ctx.queue_layout();
    let queue_size = q.size as u64;
    let chunk_size = (queue_size / 2).max(1);
    const GRANITEMOE_TINY_STEP_AQL_COUNT: u64 = 209; // from the AQL scan of this step.
    let spans_multiple_chunks = GRANITEMOE_TINY_STEP_AQL_COUNT > chunk_size;
    eprintln!(
        "poot-rocm-runtime: real HSA queue_size={queue_size} chunk_size(queue_size/2)={chunk_size} \
         granitemoe_tiny_step_aql_count={GRANITEMOE_TINY_STEP_AQL_COUNT} spans_multiple_chunks={spans_multiple_chunks}"
    );
    tracing::info!(
        queue_size,
        chunk_size,
        granitemoe_tiny_step_aql_count = GRANITEMOE_TINY_STEP_AQL_COUNT,
        spans_multiple_chunks,
        "poot-rocm-runtime: replay_graph_batched_chunk_size_vs_granitemoe_step_aql_count"
    );
    // No fixed assertion: this is a fact-finding probe. Sanity-check the size is plausible (HSA
    // requires a power-of-two queue size > 0).
    assert!(
        queue_size > 0,
        "queue_size was zero - broken queue creation?"
    );
    assert!(
        queue_size & (queue_size - 1) == 0,
        "queue_size {queue_size} is not a power of two - HSA spec violation?"
    );
}

/// Returns the byte offset of the first divergence between two byte slices, or `None` if
/// they are identical (including having equal length).
fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b.iter()).position(|(x, y)| x != y)
}

// Structural guards. Two wiring facts still cannot be observed model-free: `submit_and_wait` builds
// its publisher directly from the live `RocmContext` (its HSA queue, doorbell and function table),
// and queue creation records provenance only an HSA queue creation can prove. Each guard below reads
// only the one production file that owns the fact, extracts one item body, and asserts the call that
// wires it. They are kept small and named as structural; every behavior the wiring produces is a
// behavior test elsewhere in this file (the body writer, the ring writer's body writes, completion
// signal and fence effect, the publication contract table and the cooperative GWS facts). Each was
// observed red under the mutation named on it.

fn rust_item_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing signature `{signature}`"));
    let after = &source[start..];
    let brace = after
        .find('{')
        .unwrap_or_else(|| panic!("missing body for `{signature}`"));
    let mut depth = 0usize;
    for (offset, ch) in after[brace..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &after[brace..=brace + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced body for `{signature}`");
}

#[test]
fn production_body_writer_leaves_header_invalid_on_synthetic_ring() {
    let kernel = KernelHandle::for_test(0x1111_2222_3333_4444, 16, 32, 64, &[]);
    let mut ring = [bindings::hsa_kernel_dispatch_packet_t {
        header: 0xFFFF,
        setup: 0,
        workgroup_size_x: 0,
        workgroup_size_y: 0,
        workgroup_size_z: 0,
        reserved0: 0,
        grid_size_x: 0,
        grid_size_y: 0,
        grid_size_z: 0,
        private_segment_size: 0,
        group_segment_size: 0,
        kernel_object: 0,
        kernarg_address: std::ptr::null_mut(),
        reserved2: 0,
        completion_signal: bindings::hsa_signal_t { handle: 0 },
    }; 2];
    let kernarg = 0x55u8 as *mut c_void;
    let completion = bindings::hsa_signal_t { handle: 7 };

    // SAFETY: `ring` has two slots, slot 1 is in bounds, and no packet processor reads it.
    unsafe {
        write_kernel_dispatch_packet_body(
            ring.as_mut_ptr(),
            1,
            &kernel,
            kernarg,
            (8, 1, 1),
            (4, 1, 1),
            completion,
        );
    }

    assert_eq!(
        ring[1].header, AQL_PACKET_HEADER_INVALID,
        "body writes must leave the header INVALID until reverse header publication"
    );
    assert_ne!(ring[1].header, 0xFFFF);
    assert_eq!(ring[1].kernel_object, kernel.kernel_object());
    assert_eq!(ring[1].group_segment_size, kernel.group_segment_size());
    assert_eq!(ring[1].private_segment_size, 256);
    assert_eq!(ring[1].workgroup_size_x, 4);
    assert_eq!(ring[1].grid_size_x, 8);
    assert_eq!(ring[1].setup, 1);
    assert_eq!(ring[1].kernarg_address, kernarg);
    assert_eq!(ring[1].completion_signal.handle, 7);
    assert_eq!(
        ring[0].header, 0xFFFF,
        "unrelated slots must stay untouched"
    );
}

/// A model-free kernarg for [`AqlRingWriter`]: a fixed synthetic address, so a test can build
/// dispatches without HSA memory. A `RocmBuffer` cannot be built here at all (its `Drop` frees the
/// allocation through a live `ResourceOwner`), and the ring writer needs nothing but the address
/// (plan 558-648).
struct SyntheticKernarg(*mut c_void);

impl KernargAddress for SyntheticKernarg {
    fn kernarg_address(&self) -> *mut c_void {
        self.0
    }
}

/// A synthetic ring of `slots` packets with every header poisoned `0xFFFF`, so a test sees exactly
/// which slots a writer touched and which it left alone.
fn synthetic_ring(slots: usize) -> Vec<bindings::hsa_kernel_dispatch_packet_t> {
    vec![
        bindings::hsa_kernel_dispatch_packet_t {
            header: 0xFFFF,
            setup: 0,
            workgroup_size_x: 0,
            workgroup_size_y: 0,
            workgroup_size_z: 0,
            reserved0: 0,
            grid_size_x: 0,
            grid_size_y: 0,
            grid_size_z: 0,
            private_segment_size: 0,
            group_segment_size: 0,
            kernel_object: 0,
            kernarg_address: std::ptr::null_mut(),
            reserved2: 0,
            completion_signal: bindings::hsa_signal_t { handle: 0 },
        };
        slots
    ]
}

/// Card 617 SC-001 (behavior; replaces the source-reading
/// `structural_live_sink_writes_bodies_through_the_shared_writer`): the ring writer's `write_body`
/// publishes through the shared `write_kernel_dispatch_packet_body`, so the packet on a synthetic
/// ring carries that writer's kernel object, segment sizes, dims, kernarg address and completion
/// signal, with the header left `INVALID`. Mutation observed red: `write_body` builds its own
/// `hsa_kernel_dispatch_packet_t` (with `setup: 0`).
#[test]
fn ring_writer_write_body_uses_the_shared_body_writer() {
    let kernel = KernelHandle::for_test(0x1111_2222_3333_4444, 16, 32, 64, &[]);
    let mut ring = synthetic_ring(2);
    let kernarg_address = 0x55u8 as *mut c_void;
    let dispatches = [(
        kernel.clone(),
        SyntheticKernarg(kernarg_address),
        (8, 1, 1),
        (4, 1, 1),
    )];
    let mut writer = AqlRingWriter {
        dispatches: &dispatches,
        completion: bindings::hsa_signal_t { handle: 7 },
        ring: ring.as_mut_ptr(),
        header: 0,
    };

    writer.write_body(0, 1, true);

    assert_eq!(
        ring[1].header, AQL_PACKET_HEADER_INVALID,
        "body writes must leave the header INVALID until reverse header publication"
    );
    assert_ne!(ring[1].header, 0xFFFF);
    assert_eq!(ring[1].kernel_object, kernel.kernel_object());
    assert_eq!(ring[1].group_segment_size, kernel.group_segment_size());
    assert_eq!(ring[1].private_segment_size, 256);
    assert_eq!(ring[1].setup, 1);
    assert_eq!(ring[1].workgroup_size_x, 4);
    assert_eq!(ring[1].workgroup_size_y, 1);
    assert_eq!(ring[1].workgroup_size_z, 1);
    assert_eq!(ring[1].grid_size_x, 8);
    assert_eq!(ring[1].grid_size_y, 1);
    assert_eq!(ring[1].grid_size_z, 1);
    assert_eq!(ring[1].kernarg_address, kernarg_address);
    assert_eq!(ring[1].completion_signal.handle, 7);
    assert_eq!(
        ring[0].header, 0xFFFF,
        "unrelated slots must stay untouched"
    );
}

/// Card 617 SC-003 (behavior): only the final packet carries the completion signal - `write_body`'s
/// own choice, invisible to the free-function test above, which passes its signal on every call.
/// Mutation observed red: `write_body` passes `completion` to non-final packets too.
#[test]
fn ring_writer_completion_signal_rides_only_the_final_packet() {
    let kernel = KernelHandle::for_test(0x1111_2222_3333_4444, 16, 32, 64, &[]);
    let mut ring = synthetic_ring(2);
    let dispatches = [
        (
            kernel.clone(),
            SyntheticKernarg(0x55u8 as *mut c_void),
            (8, 1, 1),
            (4, 1, 1),
        ),
        (
            kernel,
            SyntheticKernarg(0x66u8 as *mut c_void),
            (8, 1, 1),
            (4, 1, 1),
        ),
    ];
    let mut writer = AqlRingWriter {
        dispatches: &dispatches,
        completion: bindings::hsa_signal_t { handle: 7 },
        ring: ring.as_mut_ptr(),
        header: 0,
    };

    writer.write_body(0, 0, false);
    writer.write_body(1, 1, true);

    assert_eq!(
        ring[0].completion_signal.handle, 0,
        "a non-final packet must carry the null completion signal"
    );
    assert_eq!(
        ring[1].completion_signal.handle, 7,
        "the final packet must carry the completion signal"
    );
}

/// Card 617 SC-002 (behavior; replaces the `device_store_fence` half of the source-reading
/// `structural_replay_publishes_through_the_retained_contract_and_issues_the_fence`): the ring
/// writer's `device_store_fence` issues the real architecture fence through the injected effect with
/// no live queue, and the cumulative counter rises by exactly one. Nextest runs each test in its own
/// process, so the counter is not raced here. Mutation observed red: the method drops its
/// `fence.issue()` call.
#[cfg(target_arch = "x86_64")]
#[test]
fn ring_writer_device_store_fence_issues_the_architecture_fence() {
    let kernel = KernelHandle::for_test(0x1111_2222_3333_4444, 16, 32, 64, &[]);
    let mut ring = synthetic_ring(1);
    let dispatches = [(
        kernel,
        SyntheticKernarg(0x55u8 as *mut c_void),
        (1, 1, 1),
        (1, 1, 1),
    )];
    let mut writer = AqlRingWriter {
        dispatches: &dispatches,
        completion: bindings::hsa_signal_t { handle: 0 },
        ring: ring.as_mut_ptr(),
        header: 0,
    };
    let fences_before = publication::issued_device_store_fence_count();

    writer.device_store_fence(DeviceStoreFence::X86StoreFence);

    assert_eq!(
        publication::issued_device_store_fence_count() - fences_before,
        1,
        "one device_store_fence call must issue exactly one architecture fence"
    );
}

/// STRUCTURAL GUARD. `submit_and_wait` needs a live `RocmContext` (its HSA queue, doorbell and
/// function table) and every model-free input yields the same contract either way, so the wiring is
/// read from source: the publisher behind `replay_graph_batched` publishes through the contract the
/// context retained, never a hard-coded one. The fence half of this guard became the behavior test
/// `ring_writer_device_store_fence_issues_the_architecture_fence` (card 617). Mutation observed red:
/// `submit_and_wait` validates a fresh `QueuePublication` from hard-coded `QueueRingMemory` facts.
#[test]
fn structural_replay_publishes_through_the_retained_contract() {
    let replay = rust_item_body(include_str!("hsaco.rs"), "fn submit_and_wait(");
    assert!(
        replay.contains(".queue_publication") && replay.contains("publish_chunks"),
        "replay must consume the retained context publication contract"
    );
    assert!(
        !replay.contains("QueuePublication::validate")
            && !replay.contains("QueueRingMemory::")
            && !replay.contains("QueuePublicationContract::"),
        "replay must not hard-code a publication contract in place of the retained field"
    );
}

#[test]
fn cooperative_gws_construction_records_host_coherent_system_memory_provenance() {
    assert_eq!(
        cooperative_gws_publication_facts(),
        (Some(QueueRingMemory::HostCoherent), None),
        "current cooperative GWS creator has explicit system-memory ring provenance"
    );
    assert_ne!(
        cooperative_gws_publication_facts(),
        (
            Some(QueueRingMemory::DeviceMemory),
            Some(QueueHostLink::Coherent)
        ),
        "must not silently claim coherent device-memory provenance for the GWS queue"
    );

    let validated = publication::QueuePublication::validate(
        cooperative_gws_publication_facts().0,
        cooperative_gws_publication_facts().1,
        QueueHostTarget::Other("test-host"),
    )
    .expect("host-coherent facts must validate");
    assert_eq!(validated.contract(), QueuePublicationContract::HostCoherent);
}

/// STRUCTURAL GUARD. Queue creation needs HSA, so which facts each creation plan records is read
/// from `new_inner`: the cooperative GWS plan takes its provenance from
/// `cooperative_gws_publication_facts` (tested above) and never hard-codes the device-ring facts the
/// receipt-only plan owns. Mutation observed red: the `CooperativeGws` arm records
/// `QueueRingMemory::DeviceMemory` directly.
#[test]
fn structural_gws_queue_creation_takes_its_facts_from_the_provenance_helper() {
    let construction = rust_item_body(include_str!("transfer.rs"), "fn new_inner(");
    assert!(
        construction.contains("cooperative_gws_publication_facts()"),
        "queue construction must take publication facts from the cooperative GWS provenance helper"
    );
    assert!(
        construction.contains("QueueCreationPlan::CooperativeGws")
            && construction.contains("QueueCreationPlan::PcieDeviceMemoryRingReceipt"),
        "construction must keep production GWS and receipt-only device-ring plans distinct"
    );
    let gws_arm_start = construction
        .find("QueueCreationPlan::CooperativeGws")
        .expect("CooperativeGws arm missing");
    let receipt_arm_start = construction
        .find("QueueCreationPlan::PcieDeviceMemoryRingReceipt")
        .expect("PcieDeviceMemoryRingReceipt arm missing");
    let gws_arm = &construction[gws_arm_start..receipt_arm_start];
    assert!(
        gws_arm.contains("cooperative_gws_publication_facts()")
            && !gws_arm.contains("QueueRingMemory::DeviceMemory")
            && !gws_arm.contains("QueueHostLink::Pcie"),
        "cooperative GWS arm must not hard-code device-ring publication facts"
    );
}

/// Card 522 SC-004: `RocmContext::device_caps` compares against a direct HSA query for every field HSA
/// can answer (the ISA-derived tensor-core family, the pool-size VRAM total), and against the
/// documented vendor-scoped default for the two HSA does not expose at all (LDS/group-segment size,
/// per-dimension grid count) or does not need on this backend (the watchdog budget and the card-095
/// RADV miscompile ceiling, both wgpu/RADV codegen-path defects this backend does not share). Skips
/// cleanly if ROCm is not installed.
#[test]
fn device_caps_matches_a_direct_hsa_query() {
    let ctx = match RocmContext::new() {
        Ok(ctx) => ctx,
        Err(RocmError::LibraryNotFound(msg)) => {
            tracing::info!(
                "poot-rocm-runtime: SKIP device_caps_matches_a_direct_hsa_query (no HSA library): {msg}"
            );
            return;
        }
        Err(RocmError::NoGpuAgent) => {
            tracing::info!(
                "poot-rocm-runtime: SKIP device_caps_matches_a_direct_hsa_query (no GPU agent)"
            );
            return;
        }
        Err(e) => panic!("device_caps_matches_a_direct_hsa_query failed: {e}"),
    };
    let caps = ctx.device_caps();

    // Fresh, independent HSA queries (not the context's own cached `isa_name`/`vram_total_bytes`
    // fields), against the same agent this context picked.
    // SAFETY: discovery contract: the context keeps its runtime initialized and `ctx.gpu_agent` is the
    // agent this context was built from.
    let isa = unsafe { agent_isa_name(&ctx.hsa.funcs, ctx.gpu_agent) }.expect("agent_isa_name");
    assert_eq!(
        caps.tensor_core,
        poot_target::TensorCoreSupport::from_device_name(&isa),
        "tensor_core must be classified from a fresh ISA-name query, not a hardcoded family"
    );

    let budget = ctx
        .vram_budget_bytes()
        .expect("HSA_AMD_MEMORY_POOL_INFO_SIZE sum should return a value on this device");
    assert_eq!(
        caps.max_buffer_bytes, budget,
        "max_buffer_bytes must be the queried coarse-pool total, not a hardcoded ceiling"
    );

    // HSA exposes no group-segment (LDS) size or per-dimension grid-count query this crate's discovery
    // layer has found, so these two are the documented vendor-scoped default, not a live query.
    let defaults = poot_target::DeviceCaps::rocm_default();
    assert_eq!(caps.lds_bytes, defaults.lds_bytes);
    assert_eq!(caps.max_grid, defaults.max_grid);

    // Neither the display watchdog nor the card-095 tiled-GEMM miscompile is a ROCm/HSA defect.
    assert!(caps.watchdog_budget.is_none());
    assert!(caps.known_miscompiles.tiled_gemm_max_workgroups.is_none());
}

/// Compile `body` for `ctx`'s real device arch through the same production path every real caller uses
/// (`poot_codegen::compile` + `kernel_handle`), then load it (card 608).
fn compile_and_load(
    ctx: &RocmContext,
    name: &str,
    body: &poot_kernel_ir::Body,
) -> (HsacoModule, poot_runtime_common::CompiledKernel) {
    let arch =
        poot_target::AmdArch::from_isa_name(ctx.isa_name(), ctx.wavefront()).unwrap_or_else(|e| {
            panic!(
                "{name}: could not classify device arch {:?}: {e}",
                ctx.isa_name()
            )
        });
    let target = poot_codegen::Target::AmdGcn(arch);
    let dir = std::env::temp_dir().join(format!(
        "poot-rocm-runtime-sc-fixture-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture artifact dir");
    let out = poot_codegen::artifact_path(&dir, name, target);
    poot_codegen::compile(body, target, &out).unwrap_or_else(|e| panic!("compile {name}: {e}"));
    let bytes = std::fs::read(&out).unwrap_or_else(|e| panic!("read compiled {name}: {e}"));
    let kernel = poot_codegen::kernel_handle(body, target, bytes);
    let module = ctx
        .load_hsaco(&kernel)
        .unwrap_or_else(|e| panic!("load_hsaco {name}: {e}"));
    (module, kernel)
}

/// Write an `add`-shaped kernarg: 3 `(ptr: u64, len: u64)` pairs (a, b, c), matching `add_kernel`'s
/// declaration order.
fn write_add_kernarg(
    ctx: &RocmContext,
    kernarg: &RocmBuffer,
    a: &RocmBuffer,
    b: &RocmBuffer,
    c: &RocmBuffer,
) {
    let mut karg = vec![0u8; kernarg.byte_capacity()];
    let pair = |karg: &mut [u8], slot: usize, ptr: u64, len: u64| {
        let off = slot * 16;
        karg[off..off + 8].copy_from_slice(&ptr.to_le_bytes());
        karg[off + 8..off + 16].copy_from_slice(&len.to_le_bytes());
    };
    pair(&mut karg, 0, a.device_ptr(), a.elem_count() as u64);
    pair(&mut karg, 1, b.device_ptr(), b.elem_count() as u64);
    pair(&mut karg, 2, c.device_ptr(), c.elem_count() as u64);
    ctx.write_raw_bytes(kernarg, 0, &karg)
        .expect("write add kernarg");
}

/// Card 608, SC-002 (ROCm, on the production dispatch path): a
/// real dispatch of the real `add` fixture through `RocmContext::dispatch` rejects a bound-buffer element
/// list that does not meet the compiled kernel's schema, before any submission - a same-length but
/// wrong-representation swap this check exists to catch. Mutation (recorded separately, not by this test):
/// short-circuit `check_element_schema` in `RocmContext::dispatch`/`replay_graph_batched` to always `Ok`;
/// this row would then observe `Ok(())` from the bad-schema call instead of `Err(KernelArgs(..))`.
#[test]
fn dispatch_rejects_buffers_that_do_not_meet_the_compiled_schema() {
    let Some(ctx) =
        open_context_or_skip("dispatch_rejects_buffers_that_do_not_meet_the_compiled_schema")
    else {
        return;
    };
    let body = poot_kernel_ir::fixtures::add_kernel();
    let (_module, compiled) = compile_and_load(&ctx, "add", &body);
    let kernel = ctx
        .lookup_kernel(&_module, compiled.entry_point())
        .expect("lookup add");

    let a = ctx
        .upload_f32(&[1.0, 2.0, 3.0, 4.0], BufferRole::Input)
        .expect("upload a");
    let b = ctx
        .upload_f32(&[10.0, 20.0, 30.0, 40.0], BufferRole::Input)
        .expect("upload b");
    let c = ctx.allocate_f32(4, BufferRole::Output).expect("allocate c");
    let kernarg = ctx
        .allocate_kernarg(kernel.kernarg_size() as usize)
        .expect("allocate kernarg");
    write_add_kernarg(&ctx, &kernarg, &a, &b, &c);
    let grid = (4, 1, 1);
    let block = (4, 1, 1);

    // Red: schema declares 3 F32 args; a 2-element list (wrong count) must be refused before submission.
    let wrong_count = ctx.dispatch(
        &kernel,
        &kernarg,
        &[BufferElement::F32, BufferElement::F32],
        grid,
        block,
    );
    assert!(
        matches!(
            wrong_count,
            Err(RocmError::KernelArgs(
                poot_runtime_common::KernelArgError::Count {
                    expected: 3,
                    actual: 2,
                }
            ))
        ),
        "got {wrong_count:?}"
    );

    // Red: a same-length but wrong-representation swap (F32 -> Bf16, a genuine width mismatch) at index 2.
    let wrong_kind = ctx.dispatch(
        &kernel,
        &kernarg,
        &[BufferElement::F32, BufferElement::F32, BufferElement::Bf16],
        grid,
        block,
    );
    assert!(
        matches!(
            wrong_kind,
            Err(RocmError::KernelArgs(
                poot_runtime_common::KernelArgError::ElementKind {
                    index: 2,
                    expected: BufferElement::F32,
                    actual: BufferElement::Bf16,
                }
            ))
        ),
        "got {wrong_kind:?}"
    );

    // Green: the real schema, dispatched, computes the real a+b on the device.
    ctx.dispatch(
        &kernel,
        &kernarg,
        &[BufferElement::F32, BufferElement::F32, BufferElement::F32],
        grid,
        block,
    )
    .expect("dispatch with the real schema must succeed");
    let mut got = [0.0f32; 4];
    ctx.download_f32(&c, &mut got).expect("download c");
    assert_eq!(got, [11.0, 22.0, 33.0, 44.0]);
}

/// Card 608, SC-003 (ROCm): the handle's entry point selects exactly one compiled
/// kernel. `load_hsaco` documents "one kernel per module" (ROCm has no artifact format this workspace
/// produces holding two kernels in one loadable module, so the literal "the module's first entry
/// runs" fixture does not apply here); instead this proves the same underlying claim SC-003 asks for -
/// dropping/mismatching the entry point does not let a caller silently run the wrong compiled code, it is
/// rejected by `lookup_kernel` - and that each of two distinct real fixture kernels, looked up under its
/// own entry point, dispatches and computes its own real result. Mutation (recorded separately): have
/// `lookup_kernel` skip `kernel_symbol_matches`; the mismatched lookup would then return `Ok` instead of
/// `Err`.
#[test]
fn lookup_kernel_rejects_a_mismatched_entry_point_and_dispatches_the_right_kernel() {
    let Some(ctx) = open_context_or_skip(
        "lookup_kernel_rejects_a_mismatched_entry_point_and_dispatches_the_right_kernel",
    ) else {
        return;
    };
    let (add_module, add_compiled) =
        compile_and_load(&ctx, "add", &poot_kernel_ir::fixtures::add_kernel());
    let (square_module, square_compiled) = compile_and_load(
        &ctx,
        "square",
        &poot_test_util::kernel_fixtures::square_kernel(),
    );

    // Red: add's module does not contain square's entry point (and vice versa) - rejected, not
    // silently substituted.
    let mismatched = ctx.lookup_kernel(&add_module, square_compiled.entry_point());
    assert!(
        matches!(mismatched, Err(RocmError::Hsa(_))),
        "got {mismatched:?}"
    );
    let mismatched_reverse = ctx.lookup_kernel(&square_module, add_compiled.entry_point());
    assert!(
        matches!(mismatched_reverse, Err(RocmError::Hsa(_))),
        "got {mismatched_reverse:?}"
    );

    // Green: each kernel's own entry point resolves and dispatches its own real compiled code.
    let add_kernel = ctx
        .lookup_kernel(&add_module, add_compiled.entry_point())
        .expect("lookup add under its own entry point");
    let a = ctx
        .upload_f32(&[1.0, 2.0, 3.0], BufferRole::Input)
        .expect("upload a");
    let b = ctx
        .upload_f32(&[10.0, 20.0, 30.0], BufferRole::Input)
        .expect("upload b");
    let c = ctx.allocate_f32(3, BufferRole::Output).expect("allocate c");
    let add_kernarg = ctx
        .allocate_kernarg(add_kernel.kernarg_size() as usize)
        .expect("allocate add kernarg");
    write_add_kernarg(&ctx, &add_kernarg, &a, &b, &c);
    ctx.dispatch(
        &add_kernel,
        &add_kernarg,
        &[BufferElement::F32, BufferElement::F32, BufferElement::F32],
        (3, 1, 1),
        (3, 1, 1),
    )
    .expect("dispatch add");
    let mut got_c = [0.0f32; 3];
    ctx.download_f32(&c, &mut got_c).expect("download c");
    assert_eq!(got_c, [11.0, 22.0, 33.0]);

    let square_kernel = ctx
        .lookup_kernel(&square_module, square_compiled.entry_point())
        .expect("lookup square under its own entry point");
    let x = ctx
        .upload_f32(&[2.0, 3.0, 4.0], BufferRole::Input)
        .expect("upload x");
    let y = ctx.allocate_f32(3, BufferRole::Output).expect("allocate y");
    let square_kernarg = ctx
        .allocate_kernarg(square_kernel.kernarg_size() as usize)
        .expect("allocate square kernarg");
    let mut karg = vec![0u8; square_kernarg.byte_capacity()];
    karg[0..8].copy_from_slice(&x.device_ptr().to_le_bytes());
    karg[8..16].copy_from_slice(&(x.elem_count() as u64).to_le_bytes());
    karg[16..24].copy_from_slice(&y.device_ptr().to_le_bytes());
    karg[24..32].copy_from_slice(&(y.elem_count() as u64).to_le_bytes());
    ctx.write_raw_bytes(&square_kernarg, 0, &karg)
        .expect("write square kernarg");
    ctx.dispatch(
        &square_kernel,
        &square_kernarg,
        &[BufferElement::F32, BufferElement::F32],
        (3, 1, 1),
        (3, 1, 1),
    )
    .expect("dispatch square");
    let mut got_y = [0.0f32; 3];
    ctx.download_f32(&y, &mut got_y).expect("download y");
    assert_eq!(got_y, [4.0, 9.0, 16.0]);
}

// --- card 628: BinaryOpNoContract on real ROCm/HSA hardware ------------------------------------------

/// `n` `(a, b, c, d)` rows where `a*b - c*d` computed with two separate f32 roundings differs from the
/// same expression with the second multiply and the subtract fused into one rounding step - i.e.
/// genuinely FMA-sensitive rows, found by a small deterministic search (splitmix64). Mirrors
/// `poot-runtime`'s `no_contract_tests::fma_sensitive_rows` (kept crate-local rather than shared, matching
/// this file's existing self-contained-fixture style).
fn no_contract_fma_sensitive_rows(n: usize) -> Vec<(f32, f32, f32, f32)> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next_unit = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
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

/// Write a `no_contract_probe`-shaped kernarg: 5 `(ptr: u64, len: u64)` pairs (a, b, c, d, out), matching
/// `no_contract_probe_kernel`'s declaration order.
fn write_no_contract_kernarg(
    ctx: &RocmContext,
    kernarg: &RocmBuffer,
    a: &RocmBuffer,
    b: &RocmBuffer,
    c: &RocmBuffer,
    d: &RocmBuffer,
    out: &RocmBuffer,
) {
    let mut karg = vec![0u8; kernarg.byte_capacity()];
    let pair = |karg: &mut [u8], slot: usize, ptr: u64, len: u64| {
        let off = slot * 16;
        karg[off..off + 8].copy_from_slice(&ptr.to_le_bytes());
        karg[off + 8..off + 16].copy_from_slice(&len.to_le_bytes());
    };
    pair(&mut karg, 0, a.device_ptr(), a.elem_count() as u64);
    pair(&mut karg, 1, b.device_ptr(), b.elem_count() as u64);
    pair(&mut karg, 2, c.device_ptr(), c.elem_count() as u64);
    pair(&mut karg, 3, d.device_ptr(), d.elem_count() as u64);
    pair(&mut karg, 4, out.device_ptr(), out.elem_count() as u64);
    ctx.write_raw_bytes(kernarg, 0, &karg)
        .expect("write no_contract_probe kernarg");
}

/// Dispatch `no_contract_probe_kernel(marked)` on real ROCm/HSA hardware over `rows`, returning the
/// downloaded `out` buffer.
fn dispatch_no_contract_probe(
    ctx: &RocmContext,
    marked: bool,
    rows: &[(f32, f32, f32, f32)],
) -> Vec<f32> {
    let tag = if marked { "marked" } else { "unmarked" };
    let (module, compiled) = compile_and_load(
        ctx,
        &format!("no_contract_{tag}"),
        &poot_test_util::kernel_fixtures::no_contract_probe_kernel(marked),
    );
    let kernel = ctx
        .lookup_kernel(&module, compiled.entry_point())
        .unwrap_or_else(|e| panic!("lookup no_contract_{tag}: {e}"));
    let a: Vec<f32> = rows.iter().map(|r| r.0).collect();
    let b: Vec<f32> = rows.iter().map(|r| r.1).collect();
    let c: Vec<f32> = rows.iter().map(|r| r.2).collect();
    let d: Vec<f32> = rows.iter().map(|r| r.3).collect();
    let a_buf = ctx.upload_f32(&a, BufferRole::Input).expect("upload a");
    let b_buf = ctx.upload_f32(&b, BufferRole::Input).expect("upload b");
    let c_buf = ctx.upload_f32(&c, BufferRole::Input).expect("upload c");
    let d_buf = ctx.upload_f32(&d, BufferRole::Input).expect("upload d");
    let out_buf = ctx
        .allocate_f32(rows.len(), BufferRole::Output)
        .expect("allocate out");
    let kernarg = ctx
        .allocate_kernarg(kernel.kernarg_size() as usize)
        .expect("allocate kernarg");
    write_no_contract_kernarg(ctx, &kernarg, &a_buf, &b_buf, &c_buf, &d_buf, &out_buf);
    let n = rows.len() as u32;
    ctx.dispatch(
        &kernel,
        &kernarg,
        &[
            BufferElement::F32,
            BufferElement::F32,
            BufferElement::F32,
            BufferElement::F32,
            BufferElement::F32,
        ],
        (n, 1, 1),
        (n, 1, 1),
    )
    .unwrap_or_else(|e| panic!("dispatch no_contract_{tag}: {e}"));
    let mut got = vec![0.0f32; rows.len()];
    ctx.download_f32(&out_buf, &mut got)
        .unwrap_or_else(|e| panic!("download no_contract_{tag}: {e}"));
    got
}

/// Card 628: the AMDGCN counterpart of `poot-runtime`'s `no_contract_tests` (RADV/ACO) -
/// the marked packed-dequant-style `a*b - c*d` kernel dispatched on real ROCm/HSA hardware (gfx1151),
/// checked against the unfused scalar reference bit for bit on FMA-sensitive rows. AMDGCN's compiled HSACO
/// is the device's final ISA (ahead-of-time, no further JIT), so a passing dispatch is a complete
/// real-hardware proof the constrained-intrinsic mechanism (`emit_binop_no_contract`'s Nvptx/AmdGcn branch)
/// actually blocks contraction on real LLVM-AMDGPU-backend-compiled code, not just in the disassembly-text
/// check in `poot-codegen/tests/no_contract.rs`.
///
/// Mutation (performed manually against real hardware for the card 628 report, not left in the tree -
/// `emit_binop`'s plain float Add/Sub/Mul path on AmdGcn gained an unconditional `contract` flag,
/// simulating an adversarial/future-default producer): this row stayed green (the marked subtract's
/// constrained intrinsic - a `STRICT_FSUB` DAG node, not the `ISD::FSUB` the FMA-combine pass matches on -
/// refused to fuse with the now-contract-eligible multiply feeding it), while
/// [`no_contract_unmarked_probe_matches_the_unfused_reference_on_amdgcn_today`] went red under the same
/// mutation (see its own doc comment) - real-ISA proof the marker is what's protecting this row, not
/// AMDGCN's today-happens-to-be-safe default.
#[test]
fn no_contract_marker_matches_unfused_reference_on_fma_sensitive_rows() {
    let Some(ctx) =
        open_context_or_skip("no_contract_marker_matches_unfused_reference_on_fma_sensitive_rows")
    else {
        return;
    };
    let rows = no_contract_fma_sensitive_rows(64);
    let want: Vec<f32> = rows.iter().map(|&(a, b, c, d)| a * b - c * d).collect();
    let got = dispatch_no_contract_probe(&ctx, true, &rows);
    assert_eq!(
        got, want,
        "the NoContraction-marked AMDGCN kernel diverged from the unfused scalar reference on an \
         FMA-sensitive row - the AMDGPU backend contracted the marked subtract with a producing multiply \
         despite the constrained intrinsic"
    );
}

/// Green today (unlike RADV/ACO): with `no_contract_probe_kernel(false)` (the unmarked body, compiled via
/// the same production `emit_binop`/`binop_inst` path every other kernel uses), this row does not diverge,
/// because the AMDGPU backend does not spontaneously form an FMA without an explicit `contract` fast-math
/// flag present on *both* the producing multiply and the consuming subtract (checked directly against this
/// box's toolchain: a plain, unflagged `fmul`+`fsub` pair stays two instructions on gfx1151; a
/// `contract`-flagged *consumer* alone, producer left plain, still does not fuse either - only flagging
/// both fuses it into `s_fmac_f32`).
///
/// Mutation (performed manually against real hardware for the card 628 report, not left in the tree):
/// `emit_binop` gained an unconditional `contract` flag for every plain float Add/Sub/Mul on AmdGcn,
/// giving the producing multiply the flag it's missing today. Result: RED - this row's dispatch diverged
/// from the unfused reference on most of the 64 rows (`s_fmac_f32` now formed), confirming the
/// contraction risk is real for this backend too, not merely theoretical, once a producer is contract-eligible -
/// e.g. row 0 read back `0.38405302` fused vs `0.3840531` unfused. Restored (fresh
/// `POOT_KERNEL_CACHE_DIR`, confirmed green again) before commit.
#[test]
fn no_contract_unmarked_probe_matches_the_unfused_reference_on_amdgcn_today() {
    let Some(ctx) = open_context_or_skip(
        "no_contract_unmarked_probe_matches_the_unfused_reference_on_amdgcn_today",
    ) else {
        return;
    };
    let rows = no_contract_fma_sensitive_rows(64);
    let want: Vec<f32> = rows.iter().map(|&(a, b, c, d)| a * b - c * d).collect();
    let got = dispatch_no_contract_probe(&ctx, false, &rows);
    assert_eq!(
        got, want,
        "the unmarked AMDGCN kernel diverged from the unfused reference: this backend's default just \
         started contracting without a `contract` flag present - re-check emit_binop_no_contract's \
         Nvptx/AmdGcn branch is still needed and still sufficient"
    );
}

/// Card 547a SC-001: `Weight`/`Activation` allocations read on the coarse-pool counters, at runtime
/// level - `RocmBuffer::pool()`, the per-allocation record `Pool::for_role` set at allocation time and
/// every role-aware write/read (`upload_bytes_role_aware`/`download_bytes_role_aware`) dispatches on.
///
/// An independent hardware cross-check was tried first (`vram_used_bytes`, HSA's device-wide
/// `MEMORY_AVAIL` query) and found non-discriminating on this box: Strix Halo is a unified-memory APU
/// where both pools alias the same physical GTT, so a `Fine`-pool allocation moves `MEMORY_AVAIL` by
/// the same amount a `Coarse`-pool one would (measured: a 64 MiB `Activation` allocation moved it by
/// the full 64 MiB even with the mutation below applied, routing `Activation` to `Fine`) - consistent
/// with `RocmContext::vram_used_bytes`'s own doc ("do not use it as a system-wide free-memory signal
/// on a shared unified-memory box"). `pool()` is this crate's own runtime-level record of which
/// allocator a buffer actually used, independent of `BufferRole`'s pool-agnostic byte counter in
/// `MemoryCounters` (SC-004/SC-005's counters stay correct regardless of pool).
///
/// Mutation (recorded here, applied by hand against real hardware, never left in the tree): in
/// `Pool::for_role` (`crates/poot-rocm-runtime/src/buffer.rs`), changed
/// `BufferRole::Weight | BufferRole::Activation | BufferRole::State => Self::Coarse` to route
/// `BufferRole::Activation` to `Self::Fine` instead. Result: RED -
/// `assert_eq!(activation.pool(), Pool::Coarse)` panicked ("assertion `left == right` failed... left:
/// Fine, right: Coarse"). Reverted: GREEN.
#[test]
fn weight_and_activation_allocations_use_the_coarse_pool() {
    let Some(ctx) = open_context_or_skip("weight_and_activation_allocations_use_the_coarse_pool")
    else {
        return;
    };
    let weight = ctx
        .allocate_zeroed("test", 4, BufferStorage::f32(), BufferRole::Weight)
        .expect("weight allocation");
    let activation = ctx
        .allocate_zeroed("test", 4, BufferStorage::f32(), BufferRole::Activation)
        .expect("activation allocation");
    let input = ctx
        .allocate_zeroed("test", 4, BufferStorage::f32(), BufferRole::Input)
        .expect("input allocation");
    assert_eq!(
        weight.pool(),
        Pool::Coarse,
        "Weight must allocate from the coarse pool"
    );
    assert_eq!(
        activation.pool(),
        Pool::Coarse,
        "Activation must allocate from the coarse pool"
    );
    assert_eq!(
        input.pool(),
        Pool::Fine,
        "Input (a fine-pool role) must not move to the coarse pool"
    );
}

/// Card 547a SC-003: a BF16 buffer written and read through the storage-typed primitives
/// (`allocate_zeroed`/`write_bytes`/`read_bytes`) round-trips bit for bit, with exactly `elems * 2`
/// bytes read - never a 4-byte-lane read that would silently return garbage or truncate.
///
/// Mutation (recorded here, applied by hand against real hardware, never left in the tree): in this
/// test, requested `elems * 4` bytes (as if every element were a 4-byte f32 lane) instead of `elems *
/// 2`. Result: RED - `read_bytes` panicked before any native copy:
/// `RangeOutOfBounds { op: "read_bytes", byte_offset: 0, byte_end: 68, byte_capacity: 34 }` (17
/// elements: `34 = 17*2` is the real BF16 capacity, `68 = 17*4` the wrongly-doubled request). Reverted:
/// GREEN, exact bit-for-bit round trip at the true `elems * 2` byte count.
#[test]
fn bf16_round_trips_through_the_storage_typed_primitives_with_exactly_elems_times_2_bytes() {
    let Some(ctx) = open_context_or_skip(
        "bf16_round_trips_through_the_storage_typed_primitives_with_exactly_elems_times_2_bytes",
    ) else {
        return;
    };
    let elems = 17; // odd count: a 4-byte-lane mutation would desync the tail element.
    let want_bits: Vec<u16> = (0..elems as u16).map(|i| 0x3F00 + i).collect();
    let bytes: Vec<u8> = want_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    assert_eq!(bytes.len(), elems * 2, "fixture bytes must be elems * 2");

    let buf = ctx
        .allocate_zeroed("test", elems, BufferStorage::bf16(), BufferRole::Activation)
        .expect("bf16 allocation");
    ctx.write_bytes(&buf, &bytes).expect("write_bytes");

    let mut got = vec![0u8; elems * 2];
    ctx.read_bytes(&buf, &mut got).expect("read_bytes");
    assert_eq!(
        got.len(),
        elems * 2,
        "read_bytes must read exactly elems * 2 bytes for a bf16 buffer"
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

/// Card 547a: `RocmContext::memory()` (the public, context-owned view of the one memory
/// service - not a test-owned `MemoryCounters` handed to mock pointers) must actually reach zero once
/// every allocation it reports has dropped. Every other ROCm test that reads `ctx.memory()` only
/// asserts retention (SC-004's "poisoned/pending" half, e.g.
/// `dispatch_timeout_keeps_code_object_and_argument_allocations`); this is SC-002/SC-004's "ordinary
/// completed unload releases them" half on the real context, not a mock.
///
/// Mutation (recorded here, applied by hand against real hardware, never left in the tree; card 547a): in `RocmAlloc`'s `Drop` impl (`crates/poot-rocm-runtime/src/buffer.rs`), moved the
/// `unsafe { std::mem::ManuallyDrop::drop(&mut self.guard) }` call to run only when `self.ptr == 0`
/// (i.e. never, for a real allocation) instead of unconditionally on the non-poisoned path. Result:
/// RED - `ctx.memory()'s Activation role must read zero live bytes once every allocation it
/// references has dropped: got 320` (the two buffers' 64 + 256 bytes stayed charged after both
/// `RocmBuffer`s dropped). Reverted: GREEN.
#[test]
fn rocm_context_memory_returns_to_zero_after_every_allocation_drops() {
    let Some(ctx) =
        open_context_or_skip("rocm_context_memory_returns_to_zero_after_every_allocation_drops")
    else {
        return;
    };
    let before = ctx
        .memory()
        .into_iter()
        .find(|&(role, _)| role == BufferRole::Activation)
        .map(|(_, snap)| snap.live_bytes)
        .unwrap_or(0);

    let a = ctx
        .allocate_zeroed("test_a", 16, BufferStorage::f32(), BufferRole::Activation)
        .expect("allocate a");
    let b = ctx
        .allocate_zeroed("test_b", 64, BufferStorage::f32(), BufferRole::Activation)
        .expect("allocate b");
    let during = ctx
        .memory()
        .into_iter()
        .find(|&(role, _)| role == BufferRole::Activation)
        .map(|(_, snap)| snap.live_bytes)
        .unwrap_or(0);
    assert_eq!(
        during,
        before + a.byte_capacity() as u64 + b.byte_capacity() as u64,
        "both live allocations must be charged while held"
    );

    drop(a);
    drop(b);
    let after = ctx
        .memory()
        .into_iter()
        .find(|&(role, _)| role == BufferRole::Activation)
        .map(|(_, snap)| snap.live_bytes)
        .unwrap_or(0);
    assert_eq!(
        after, before,
        "ctx.memory()'s Activation role must read zero live bytes once every allocation it \
         references has dropped: got {after}"
    );
}
