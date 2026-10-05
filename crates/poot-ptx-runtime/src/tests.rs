use super::*;

mod require_backend_tests {
    use super::require_gpu_check_with;

    #[test]
    fn ptx_context_open_is_required_by_its_own_variable_only() {
        use poot_runtime_common::DeviceBackend;
        let fails_open = |variable: &'static str| {
            std::panic::catch_unwind(|| {
                require_gpu_check_with(|name| (name == variable).then(|| "1".into()), "no device")
            })
            .is_err()
        };
        assert!(fails_open(DeviceBackend::Ptx.variable()));
        assert!(
            !fails_open("POOT_REQUIRE_GPU"),
            "the retired all-backend switch must not require Ptx"
        );
        for other in DeviceBackend::ALL
            .into_iter()
            .filter(|other| *other != DeviceBackend::Ptx)
        {
            assert!(
                !fails_open(other.variable()),
                "{other:?} must not require Ptx"
            );
        }
    }
}

use std::cell::{Cell, RefCell};
use std::sync::Mutex;

/// A test double for [`CudaDriver`] (R471-010): `Mutex`-backed, not `Cell`/`RefCell`-backed, so it is
/// `Send + Sync` like every real implementor must be now that `Arc<dyn CudaDriver>` replaces `Rc`.
#[derive(Default)]
struct MockCleanup {
    events: Mutex<Vec<String>>,
    current: Mutex<Option<usize>>,
    fail_next_set: Mutex<bool>,
    fail_set_target: Mutex<Option<usize>>,
}

impl MockCleanup {
    fn record(&self, event: impl Into<String>) {
        self.events.lock().unwrap().push(event.into());
    }

    fn snapshot(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }

    fn clear(&self) {
        self.events.lock().unwrap().clear();
    }

    fn current(&self) -> Option<sys::CUcontext> {
        (*self.current.lock().unwrap()).map(fake_context)
    }

    fn set_mock_current(&self, ctx: Option<sys::CUcontext>) {
        *self.current.lock().unwrap() = ctx.map(|ctx| ctx as usize);
    }

    fn fail_set_to(&self, ctx: Option<sys::CUcontext>) {
        *self.fail_set_target.lock().unwrap() = Some(ctx.map_or(0, |ctx| ctx as usize));
    }

    fn context_event(&self, prefix: &str, ctx: Option<sys::CUcontext>) -> String {
        match ctx {
            Some(ctx) => format!("{prefix}:{}", ctx as usize),
            None => format!("{prefix}:none"),
        }
    }

    fn record_context_dependent(&self, event: &'static str) {
        if self.current() == Some(fake_context(11)) {
            self.record(event);
        } else {
            self.record(format!("{event}:wrong_context"));
        }
    }
}

impl CudaDriver for MockCleanup {
    fn get_current(&self) -> Result<Option<sys::CUcontext>, DriverError> {
        let current = self.current();
        self.record(self.context_event("driver:get_current", current));
        Ok(current)
    }

    fn set_current(&self, ctx: Option<sys::CUcontext>) -> Result<(), DriverError> {
        self.record(self.context_event("driver:set_current", ctx));
        let target = ctx.map_or(0, |ctx| ctx as usize);
        let fail_target = *self.fail_set_target.lock().unwrap() == Some(target);
        if fail_target {
            *self.fail_set_target.lock().unwrap() = None;
            return Err(DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN));
        }
        if std::mem::replace(&mut *self.fail_next_set.lock().unwrap(), false) {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT));
        }
        self.set_mock_current(ctx);
        Ok(())
    }

    fn free(&self, _ptr: sys::CUdeviceptr) {
        self.record_context_dependent("cleanup:free");
    }

    fn unload_module(&self, _module: sys::CUmodule) {
        self.record_context_dependent("cleanup:unload_module");
    }

    fn destroy_stream(&self, _stream: sys::CUstream) {
        self.record_context_dependent("cleanup:destroy_stream");
    }

    fn release_primary(&self, _device: sys::CUdevice) {
        if self.current() == Some(fake_context(11)) {
            self.record("cleanup:release_primary:while_current");
        } else {
            self.record("cleanup:release_primary");
        }
    }

    fn destroy_event(&self, _event: sys::CUevent) {
        self.record_context_dependent("cleanup:destroy_event");
    }

    fn destroy_graph_exec(&self, _exec: sys::CUgraphExec) {
        self.record_context_dependent("cleanup:destroy_graph_exec");
    }

    fn destroy_graph(&self, _graph: sys::CUgraph) {
        self.record_context_dependent("cleanup:destroy_graph");
    }

    fn launch_graph(
        &self,
        _exec: sys::CUgraphExec,
        _stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        self.record_context_dependent("operation:launch_graph");
        Ok(())
    }

    fn synchronize_stream(&self, _stream: sys::CUstream) -> Result<(), DriverError> {
        self.record_context_dependent("operation:synchronize_stream");
        Ok(())
    }
}

fn fake_context(value: usize) -> sys::CUcontext {
    value as sys::CUcontext
}

fn fake_stream(value: usize) -> sys::CUstream {
    value as sys::CUstream
}

fn fake_module(value: usize) -> sys::CUmodule {
    value as sys::CUmodule
}

fn fake_event(value: usize) -> sys::CUevent {
    value as sys::CUevent
}

fn fake_graph(value: usize) -> sys::CUgraph {
    value as sys::CUgraph
}

fn fake_graph_exec(value: usize) -> sys::CUgraphExec {
    value as sys::CUgraphExec
}

fn injected_error() -> PtxError {
    PtxError::BadKernel("injected lifecycle failure".to_string())
}

fn mock_owner(mock: &Arc<MockCleanup>) -> Arc<PtxOwner> {
    let driver: Arc<dyn CudaDriver> = mock.clone();
    Arc::new(PtxOwner {
        driver,
        device: 7,
        ctx: fake_context(11),
        stream: fake_stream(13),
        modules: Mutex::new(Vec::new()),
        memory: poot_runtime_common::MemoryCounters::new(),
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ContextFailure {
    Retain,
    Activation,
    Stream,
    None,
}

#[test]
fn context_construction_releases_each_acquired_handle_in_order() {
    struct Case {
        failure: ContextFailure,
        prior: Option<sys::CUcontext>,
        fail_unbind: bool,
        expected_current: Option<sys::CUcontext>,
        expected: Vec<&'static str>,
    }

    let cases = [
        Case {
            failure: ContextFailure::Retain,
            prior: None,
            fail_unbind: false,
            expected_current: None,
            expected: vec!["acquire:retain"],
        },
        Case {
            failure: ContextFailure::Activation,
            prior: None,
            fail_unbind: false,
            expected_current: None,
            expected: vec![
                "acquire:retain",
                "driver:get_current:none",
                "driver:set_current:11",
                "driver:set_current:none",
                "cleanup:release_primary",
            ],
        },
        Case {
            failure: ContextFailure::Activation,
            prior: Some(fake_context(31)),
            fail_unbind: false,
            expected_current: Some(fake_context(31)),
            expected: vec![
                "acquire:retain",
                "driver:get_current:31",
                "driver:set_current:11",
                "driver:set_current:31",
                "cleanup:release_primary",
            ],
        },
        Case {
            failure: ContextFailure::Activation,
            prior: Some(fake_context(11)),
            fail_unbind: false,
            expected_current: None,
            expected: vec![
                "acquire:retain",
                "driver:get_current:11",
                "driver:set_current:11",
                "driver:set_current:11",
                "driver:set_current:none",
                "cleanup:release_primary",
            ],
        },
        Case {
            failure: ContextFailure::Activation,
            prior: Some(fake_context(11)),
            fail_unbind: true,
            expected_current: Some(fake_context(11)),
            expected: vec![
                "acquire:retain",
                "driver:get_current:11",
                "driver:set_current:11",
                "driver:set_current:11",
                "driver:set_current:none",
            ],
        },
        Case {
            failure: ContextFailure::Stream,
            prior: None,
            fail_unbind: false,
            expected_current: None,
            expected: vec![
                "acquire:retain",
                "driver:get_current:none",
                "driver:set_current:11",
                "acquire:create_stream",
                "driver:set_current:none",
                "cleanup:release_primary",
            ],
        },
        Case {
            failure: ContextFailure::None,
            prior: None,
            fail_unbind: false,
            expected_current: None,
            expected: vec![
                "acquire:retain",
                "driver:get_current:none",
                "driver:set_current:11",
                "acquire:create_stream",
                "driver:get_current:11",
                "driver:set_current:11",
                "cleanup:destroy_stream",
                "driver:set_current:none",
                "cleanup:release_primary",
            ],
        },
    ];

    for case in cases {
        let failure = case.failure;
        let mock = Arc::new(MockCleanup::default());
        mock.set_mock_current(case.prior);
        let driver: Arc<dyn CudaDriver> = mock.clone();
        let retain_mock = Arc::clone(&mock);
        let stream_mock = Arc::clone(&mock);
        if failure == ContextFailure::Activation {
            *mock.fail_next_set.lock().unwrap() = true;
        }
        if case.fail_unbind {
            mock.fail_set_to(None);
        }
        let result = construct_owner(
            driver,
            7,
            move || {
                retain_mock.record("acquire:retain");
                if failure == ContextFailure::Retain {
                    Err(injected_error())
                } else {
                    Ok(fake_context(11))
                }
            },
            move || {
                stream_mock.record("acquire:create_stream");
                if failure == ContextFailure::Stream {
                    Err(injected_error())
                } else {
                    Ok(fake_stream(13))
                }
            },
        );
        match (failure, result) {
            (ContextFailure::None, Ok(owner)) => drop(owner),
            (
                ContextFailure::Activation,
                Err(PtxError::Driver(DriverError(sys::CUresult::CUDA_ERROR_INVALID_CONTEXT))),
            ) => {}
            (ContextFailure::Retain | ContextFailure::Stream, Err(_)) => {}
            _ => panic!("constructor returned the wrong injected result"),
        }
        assert_eq!(mock.snapshot(), case.expected);
        assert_eq!(mock.current(), case.expected_current);
    }
}

#[test]
fn cleanup_restores_foreign_and_no_current_states() {
    let cases = [
        (
            Some(fake_context(31)),
            vec![
                "driver:get_current:31",
                "driver:set_current:11",
                "cleanup:free",
                "driver:set_current:31",
            ],
        ),
        (
            None,
            vec![
                "driver:get_current:none",
                "driver:set_current:11",
                "cleanup:free",
                "driver:set_current:none",
            ],
        ),
    ];

    for (prior, expected) in cases {
        let mock = Arc::new(MockCleanup::default());
        mock.set_mock_current(prior);
        let owner = mock_owner(&mock);
        let buffer = allocate_owned(
            Arc::clone(&owner),
            "test_alloc",
            1,
            BufferStorage::f32(),
            BufferRole::Activation,
            |_| Ok(101),
            |_, _| Ok(()),
        )
        .expect("mock allocation");

        drop(buffer);

        assert_eq!(mock.snapshot(), expected);
        assert_eq!(mock.current(), prior);
        drop(owner);
    }
}

#[test]
fn cleanup_activation_failure_suppresses_context_dependent_destruction() {
    let mock = Arc::new(MockCleanup::default());
    let foreign = Some(fake_context(31));
    mock.set_mock_current(foreign);
    let owner = mock_owner(&mock);
    let buffer = allocate_owned(
        Arc::clone(&owner),
        "test_alloc",
        1,
        BufferStorage::f32(),
        BufferRole::Activation,
        |_| Ok(101),
        |_, _| Ok(()),
    )
    .expect("mock allocation");
    *mock.fail_next_set.lock().unwrap() = true;

    drop(buffer);

    assert_eq!(
        mock.snapshot(),
        [
            "driver:get_current:31",
            "driver:set_current:11",
            "driver:set_current:31",
        ]
    );
    assert_eq!(mock.current(), foreign);
    drop(owner);
}

#[test]
fn final_owner_release_unbinds_owner_or_restores_prior_context() {
    let cases = [
        (
            Some(fake_context(11)),
            None,
            vec![
                "driver:get_current:11",
                "driver:set_current:11",
                "cleanup:destroy_stream",
                "driver:set_current:none",
                "cleanup:release_primary",
            ],
        ),
        (
            Some(fake_context(31)),
            Some(fake_context(31)),
            vec![
                "driver:get_current:31",
                "driver:set_current:11",
                "cleanup:destroy_stream",
                "driver:set_current:31",
                "cleanup:release_primary",
            ],
        ),
        (
            None,
            None,
            vec![
                "driver:get_current:none",
                "driver:set_current:11",
                "cleanup:destroy_stream",
                "driver:set_current:none",
                "cleanup:release_primary",
            ],
        ),
    ];

    for (prior, restored, expected) in cases {
        let mock = Arc::new(MockCleanup::default());
        mock.set_mock_current(prior);

        drop(mock_owner(&mock));

        assert_eq!(mock.snapshot(), expected);
        assert_eq!(mock.current(), restored);
    }
}

#[test]
fn final_owner_restore_failure_suppresses_primary_release() {
    let mock = Arc::new(MockCleanup::default());
    mock.set_mock_current(Some(fake_context(31)));
    mock.fail_set_to(Some(fake_context(31)));

    drop(mock_owner(&mock));

    assert_eq!(
        mock.snapshot(),
        [
            "driver:get_current:31",
            "driver:set_current:11",
            "cleanup:destroy_stream",
            "driver:set_current:31",
        ]
    );
    assert_eq!(mock.current(), Some(fake_context(11)));
}

#[test]
fn surviving_graph_scopes_launch_and_sync_after_context_switch_or_clear() {
    let mock = Arc::new(MockCleanup::default());
    let context = PtxContext {
        owner: mock_owner(&mock),
        funcs: RefCell::new(HashMap::new()),
        capture: CaptureRetention::default(),
        device_caps: poot_target::DeviceCaps::ptx_default(),
        device_timing: false,
    };
    let graph = instantiate_graph(
        Arc::clone(&context.owner),
        fake_graph(23),
        Vec::new(),
        |exec| {
            *exec = fake_graph_exec(29);
            Ok(())
        },
    )
    .expect("mock graph");
    drop(context);

    mock.set_mock_current(Some(fake_context(31)));
    graph.launch().expect("launch under retained owner");
    assert_eq!(
        mock.snapshot(),
        [
            "driver:get_current:31",
            "driver:set_current:11",
            "operation:launch_graph",
            "driver:set_current:31",
        ]
    );
    assert_eq!(mock.current(), Some(fake_context(31)));

    mock.clear();
    mock.set_mock_current(None);
    graph
        .synchronize()
        .expect("synchronize under retained owner");
    assert_eq!(
        mock.snapshot(),
        [
            "driver:get_current:none",
            "driver:set_current:11",
            "operation:synchronize_stream",
            "driver:set_current:none",
        ]
    );
    assert_eq!(mock.current(), None);
    drop(graph);
}

#[test]
fn allocation_initialization_failure_frees_before_returning() {
    for initialization in ["initialize:copy", "initialize:memset"] {
        let mock = Arc::new(MockCleanup::default());
        let owner = mock_owner(&mock);

        let result = allocate_owned(
            Arc::clone(&owner),
            "test_allocation",
            4,
            BufferStorage::f32(),
            BufferRole::Activation,
            |_| {
                mock.record("acquire:malloc");
                Ok(101)
            },
            |_, _| {
                mock.record(initialization);
                Err(injected_error())
            },
        );

        assert!(result.is_err());
        assert_eq!(
            mock.snapshot(),
            [
                "acquire:malloc",
                initialization,
                "driver:get_current:none",
                "driver:set_current:11",
                "cleanup:free",
                "driver:set_current:none",
            ]
        );
        drop(owner);
    }
}

#[test]
fn module_lookup_failure_unloads_the_loaded_module() {
    let mock = Arc::new(MockCleanup::default());
    let owner = mock_owner(&mock);
    mock.record("acquire:load_module");

    let guard = ModuleConstruction::new(fake_module(17), Arc::clone(&owner));
    mock.record("lookup:get_function_failed");
    drop(guard);

    assert_eq!(
        mock.snapshot(),
        [
            "acquire:load_module",
            "lookup:get_function_failed",
            "driver:get_current:none",
            "driver:set_current:11",
            "cleanup:unload_module",
            "driver:set_current:none",
        ]
    );
    drop(owner);
}

#[test]
fn event_pair_failure_destroys_the_first_event() {
    let cases = [
        (0, vec!["acquire:create_event"]),
        (
            1,
            vec![
                "acquire:create_event",
                "acquire:create_event",
                "driver:get_current:none",
                "driver:set_current:11",
                "cleanup:destroy_event",
                "driver:set_current:none",
            ],
        ),
    ];

    for (fail_call, expected) in cases {
        let mock = Arc::new(MockCleanup::default());
        let owner = mock_owner(&mock);
        let calls = Cell::new(0);
        let result = create_event_pair(Arc::clone(&owner), || {
            let call = calls.get();
            calls.set(call + 1);
            mock.record("acquire:create_event");
            if call == fail_call {
                Err(injected_error())
            } else {
                Ok(fake_event(19 + call))
            }
        });

        assert!(result.is_err());
        assert_eq!(mock.snapshot(), expected);
        drop(owner);
    }
}

#[test]
fn graph_instantiate_failure_destroys_partial_exec_then_graph() {
    let cases = [
        (
            false,
            vec![
                "acquire:instantiate_graph",
                "driver:get_current:none",
                "driver:set_current:11",
                "cleanup:destroy_graph",
                "driver:set_current:none",
            ],
        ),
        (
            true,
            vec![
                "acquire:instantiate_graph",
                "driver:get_current:none",
                "driver:set_current:11",
                "cleanup:destroy_graph_exec",
                "cleanup:destroy_graph",
                "driver:set_current:none",
            ],
        ),
    ];

    for (returned_exec, expected) in cases {
        let mock = Arc::new(MockCleanup::default());
        let owner = mock_owner(&mock);
        let result = instantiate_graph(Arc::clone(&owner), fake_graph(23), Vec::new(), |exec| {
            mock.record("acquire:instantiate_graph");
            if returned_exec {
                *exec = fake_graph_exec(29);
            }
            Err(injected_error())
        });

        assert!(result.is_err());
        assert_eq!(mock.snapshot(), expected);
        drop(owner);
    }
}

#[test]
fn children_delay_context_cleanup_and_preserve_dependency_order() {
    let mock = Arc::new(MockCleanup::default());
    let owner = mock_owner(&mock);
    ModuleConstruction::new(fake_module(17), Arc::clone(&owner)).finish();
    let context = PtxContext {
        owner,
        funcs: RefCell::new(HashMap::new()),
        capture: CaptureRetention::default(),
        device_caps: poot_target::DeviceCaps::ptx_default(),
        device_timing: false,
    };
    let buffer = allocate_owned(
        Arc::clone(&context.owner),
        "test_alloc",
        1,
        BufferStorage::f32(),
        BufferRole::Activation,
        |_| Ok(101),
        |_, _| Ok(()),
    )
    .expect("mock allocation");
    let graph = instantiate_graph(
        Arc::clone(&context.owner),
        fake_graph(23),
        Vec::new(),
        |exec| {
            *exec = fake_graph_exec(29);
            Ok(())
        },
    )
    .expect("mock graph");

    drop(context);
    assert!(mock.snapshot().is_empty(), "children must retain the owner");
    drop(graph);
    assert_eq!(
        mock.snapshot(),
        [
            "driver:get_current:none",
            "driver:set_current:11",
            "cleanup:destroy_graph_exec",
            "cleanup:destroy_graph",
            "driver:set_current:none",
        ]
    );
    drop(buffer);
    assert_eq!(
        mock.snapshot(),
        [
            "driver:get_current:none",
            "driver:set_current:11",
            "cleanup:destroy_graph_exec",
            "cleanup:destroy_graph",
            "driver:set_current:none",
            "driver:get_current:none",
            "driver:set_current:11",
            "cleanup:free",
            "driver:set_current:none",
            "driver:get_current:none",
            "driver:set_current:11",
            "cleanup:unload_module",
            "cleanup:destroy_stream",
            "driver:set_current:none",
            "cleanup:release_primary",
        ]
    );
}

fn mock_context(mock: &Arc<MockCleanup>) -> PtxContext {
    PtxContext {
        owner: mock_owner(mock),
        funcs: RefCell::new(HashMap::new()),
        capture: CaptureRetention::default(),
        device_caps: poot_target::DeviceCaps::ptx_default(),
        device_timing: false,
    }
}

fn free_count(mock: &MockCleanup) -> usize {
    mock.snapshot()
        .iter()
        .filter(|event| *event == "cleanup:free")
        .count()
}

/// Card 516a SC-002: a captured graph owns every buffer its dispatches referenced, so dropping the caller's
/// handles cannot free memory a replay still addresses. The allocations are freed only after the graph that
/// records their addresses is destroyed.
#[test]
fn captured_graph_owns_dispatched_buffers_until_it_drops() {
    let mock = Arc::new(MockCleanup::default());
    let context = mock_context(&mock);
    let alloc = |ptr: sys::CUdeviceptr| {
        allocate_owned(
            Arc::clone(&context.owner),
            "test_alloc",
            4,
            BufferStorage::f32(),
            BufferRole::Activation,
            |_| Ok(ptr),
            |_, _| Ok(()),
        )
        .expect("mock allocation")
    };
    let (a, b, out) = (alloc(101), alloc(102), alloc(103));
    let uncaptured = alloc(104);

    context.capture.begin();
    let args = KernelArgs::pack(
        &context.capture,
        &[&a, &b],
        &[a.elem_count(), b.elem_count()],
        &out,
        out.elem_count(),
    );
    let params: Vec<u64> = args
        .params()
        .iter()
        // SAFETY: each parameter points at a live `u64` device pointer or `i64` length owned by `args`.
        .map(|param| unsafe { param.cast::<u64>().read() })
        .collect();
    assert_eq!(params, [101, 4, 102, 4, 103, 4]);
    drop((a, b, out));
    assert_eq!(
        free_count(&mock),
        0,
        "an open capture must retain the buffers it dispatched"
    );

    let graph = instantiate_graph(
        Arc::clone(&context.owner),
        fake_graph(23),
        context.capture.finish(),
        |exec| {
            *exec = fake_graph_exec(29);
            Ok(())
        },
    )
    .expect("mock graph");
    assert!(!context.capture.is_open());
    drop(uncaptured);
    assert_eq!(
        free_count(&mock),
        1,
        "only the buffer no dispatch referenced is freed while the graph lives"
    );
    graph
        .launch()
        .expect("launch with every captured buffer alive");
    assert_eq!(free_count(&mock), 1);

    mock.clear();
    drop(graph);
    let free_one = [
        "driver:get_current:none",
        "driver:set_current:11",
        "cleanup:free",
        "driver:set_current:none",
    ];
    let mut expected = vec![
        "driver:get_current:none",
        "driver:set_current:11",
        "cleanup:destroy_graph_exec",
        "cleanup:destroy_graph",
        "driver:set_current:none",
    ];
    for _ in 0..3 {
        expected.extend(free_one);
    }
    assert_eq!(mock.snapshot(), expected);
}

/// Card 516a: while a capture is open only kernel dispatches may reach the stream. Every other operation is
/// refused before any driver call, since it would allocate, free, synchronize, or record a host copy that
/// every replay would repeat from caller memory.
#[test]
fn every_non_dispatch_operation_refuses_while_a_capture_is_open() {
    type Operation = fn(&PtxContext, &PtxBuffer) -> Result<(), PtxError>;
    let operations: [(&str, Operation); 12] = [
        ("synchronize", |ctx, _| ctx.synchronize()),
        ("upload_f32", |ctx, _| ctx.upload_f32(&[1.0]).map(drop)),
        ("upload_i32", |ctx, _| ctx.upload_i32(&[1]).map(drop)),
        ("alloc_f32", |ctx, _| ctx.alloc_f32(1).map(drop)),
        ("upload_bf16_bytes", |ctx, _| {
            ctx.upload_bf16_bytes(&[0, 0]).map(drop)
        }),
        ("upload_f16", |ctx, _| ctx.upload_f16(&[0, 0]).map(drop)),
        ("update_f32", |ctx, buf| ctx.update_f32(buf, &[1.0; 4])),
        ("write_bytes", |ctx, buf| ctx.write_bytes(buf, &[0u8; 16])),
        ("read_bytes", |ctx, buf| ctx.read_bytes(buf, &mut [0u8; 16])),
        ("download_f32", |ctx, buf| ctx.download_f32(buf).map(drop)),
        ("download_i32", |ctx, buf| ctx.download_i32(buf).map(drop)),
        ("begin_capture", |ctx, _| ctx.begin_capture()),
    ];
    let mock = Arc::new(MockCleanup::default());
    let context = mock_context(&mock);
    let buffer = test_buffer(4, BufferStorage::f32());
    context.capture.begin();
    for (name, operation) in operations {
        match operation(&context, &buffer) {
            Err(PtxError::CaptureOpen { op }) => assert_eq!(op, name),
            other => panic!("{name} during capture: expected CaptureOpen, got {other:?}"),
        }
    }
    assert!(
        mock.snapshot().is_empty(),
        "a refused operation must not reach the driver"
    );
    assert!(context.capture.is_open());
}

fn test_buffer(elements: usize, storage: BufferStorage) -> PtxBuffer {
    checked_buffer(0, "test_buffer", elements, storage, None).expect("valid test buffer")
}

fn count_copy(
    buffer: &PtxBuffer,
    expected: BufferStorage,
    offset: usize,
    elements: usize,
    whole: bool,
    calls: &Cell<usize>,
) -> Result<(), PtxError> {
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
        PtxError::SizeMismatch {
            op: "test_copy",
            have: 2,
            got: 3
        }
    ));
    assert_eq!(calls.get(), 0, "rejection entered the native copy closure");
}

#[test]
fn narrow_buffer_rejected_by_f32_operation_before_copy() {
    let buffer = test_buffer(4, BufferStorage::bf16());
    let calls = Cell::new(0);

    let error = count_copy(&buffer, BufferStorage::f32(), 0, 4, true, &calls)
        .expect_err("bf16 storage must not admit an f32 copy");

    let PtxError::RepresentationMismatch {
        op: "test_copy",
        expected,
        actual,
    } = error
    else {
        panic!("expected RepresentationMismatch, got {error:?}");
    };
    assert_eq!(expected, BufferStorage::f32());
    assert_eq!(actual, BufferStorage::bf16());
    assert_eq!(calls.get(), 0, "rejection entered the native copy closure");
}

/// Card 527 (R484-001, R471-009), mechanism-level (the production bind-path acceptance evidence is
/// `poot-ptx-gpu`'s `bind_value_for_capture_bf16_cache_mismatch_is_a_typed_refusal`, card 527): `bf16_packed()` canonicalizes on `ElementKind::I32` (review F2), the one word PTX's packed
/// upload would use if it ever packed (it does not - Nvptx never selects `Bf16Packed`, but the shared
/// `poot-target` type still has to compare correctly for the backends that do). A packed-BF16 value and
/// a plain dense I32 value share native element kind; only `checked_copy`'s full `BufferStorage`
/// comparison (not element kind alone) tells them apart.
#[test]
fn bf16_packed_word_rejected_by_dense_i32_bind() {
    let buffer = test_buffer(7, BufferStorage::bf16_packed());
    let calls = Cell::new(0);

    let error = count_copy(&buffer, BufferStorage::i32(), 2, 3, false, &calls)
        .expect_err("packed bf16 lanes must not admit a dense-i32 copy");

    let PtxError::RepresentationMismatch {
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
    assert_eq!(calls.get(), 0, "rejection entered the native copy closure");
}

#[test]
fn overflowing_partial_offset_makes_zero_copy_calls() {
    let buffer = test_buffer(4, BufferStorage::f32());
    let calls = Cell::new(0);

    let error = count_copy(&buffer, BufferStorage::f32(), usize::MAX, 1, false, &calls)
        .expect_err("overflowing element offset must fail");

    assert!(matches!(
        error,
        PtxError::RangeOverflow {
            op: "test_copy",
            element_offset: usize::MAX,
            elements: 1,
        }
    ));
    assert_eq!(calls.get(), 0, "rejection entered the native copy closure");
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
        checked_layout("alloc_f32", usize::MAX, BufferStorage::f32()),
        Err(PtxError::ElementCountTooLarge { .. } | PtxError::ByteSizeOverflow { .. })
    ));
}

const LIFECYCLE_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry card305_add_one(
    .param .u64 input_ptr,
    .param .u64 input_len,
    .param .u64 output_ptr,
    .param .u64 output_len
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<2>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [input_ptr];
    ld.param.u64 %rd2, [input_len];
    ld.param.u64 %rd3, [output_ptr];
    mov.u32 %r1, %tid.x;
    cvt.u64.u32 %rd4, %r1;
    setp.ge.u64 %p1, %rd4, %rd2;
    @%p1 bra DONE;
    mul.wide.u32 %rd5, %r1, 4;
    add.s64 %rd6, %rd1, %rd5;
    add.s64 %rd7, %rd3, %rd5;
    ld.global.f32 %f1, [%rd6];
    add.f32 %f2, %f1, 0f3F800000;
    st.global.f32 [%rd7], %f2;

DONE:
    ret;
}
"#;

/// A [`CompiledKernel`] for [`LIFECYCLE_PTX`]'s one-input-one-output f32 kernel: hand-written PTX text,
/// not compiler output, so this test builds the handle directly (card 608's sanctioned "imported kernel"
/// escape hatch, see `poot_runtime_common::CompiledKernel::new`'s doc).
fn lifecycle_kernel() -> poot_runtime_common::CompiledKernel {
    // SAFETY: `LIFECYCLE_PTX` above declares exactly one f32 input pointer/length pair and one f32
    // output pointer/length pair, matching this schema; it never traps.
    unsafe {
        poot_runtime_common::CompiledKernel::new(
            poot_target::Backend::Nvptx,
            "card305_add_one",
            poot_runtime_common::KernelCode::Ptx(LIFECYCLE_PTX.into()),
            vec![
                poot_runtime_common::ArgSchema::new(
                    poot_target::ElementKind::F32,
                    poot_runtime_common::ArgAccess::Read,
                ),
                poot_runtime_common::ArgSchema::new(
                    poot_target::ElementKind::F32,
                    poot_runtime_common::ArgAccess::Write,
                ),
            ],
            false,
        )
    }
}

#[test]
fn card305_ptx_lifecycle_repeated_contexts_and_surviving_children() {
    let observer = match PtxContext::new() {
        Ok(context) => context,
        Err(error) => {
            eprintln!(
                "SKIP card305_ptx_lifecycle_repeated_contexts_and_surviving_children: {error}"
            );
            return;
        }
    };
    let expected = [2.0, 3.0, 4.0, 5.0];
    let mut free_samples = Vec::with_capacity(32);

    for cycle in 0..40 {
        let worker = PtxContext::new().expect("repeated PTX context construction");
        let input = worker
            .upload_f32(&[1.0, 2.0, 3.0, 4.0])
            .expect("lifecycle input upload");
        let output = worker.alloc_f32(4).expect("lifecycle output allocation");
        worker.synchronize().expect("settle output initialization");

        worker
            .dispatch_dev(
                "card305_lifecycle_warm",
                &lifecycle_kernel(),
                [32, 1, 1],
                [4, 1, 1],
                &[&input],
                &[input.elem_count()],
                &output,
                output.elem_count(),
            )
            .expect("load module and warm tiny kernel");
        worker.synchronize().expect("settle warm dispatch");

        worker.begin_capture().expect("begin lifecycle capture");
        worker
            .dispatch_dev(
                "card305_lifecycle_capture",
                &lifecycle_kernel(),
                [32, 1, 1],
                [4, 1, 1],
                &[&input],
                &[input.elem_count()],
                &output,
                output.elem_count(),
            )
            .expect("capture tiny kernel");
        let graph = worker.end_capture().expect("instantiate lifecycle graph");

        drop(worker);
        // A surviving graph is its own usable child, not merely a lifetime token. Clear the thread's current context
        // before both use and destruction so its retained owner must activate the originating context and
        // restore the no-current state itself.
        // SAFETY: a null context only unbinds the calling thread.
        unsafe { result::ctx::set_current(std::ptr::null_mut()) }
            .expect("clear current context before surviving graph use");
        graph
            .launch()
            .expect("launch graph after context handle drop");
        graph
            .synchronize()
            .expect("synchronize graph after context handle drop");
        assert_eq!(
            result::ctx::get_current().expect("query context after surviving graph use"),
            None,
            "surviving graph use must restore no-current state"
        );
        observer
            .make_current()
            .expect("rebind observer before output download");
        assert_eq!(
            observer
                .download_f32(&output)
                .expect("read surviving output buffer"),
            expected,
            "cycle {cycle}"
        );

        // SAFETY: a null context only unbinds the calling thread.
        unsafe { result::ctx::set_current(std::ptr::null_mut()) }
            .expect("clear current context before surviving child drops");
        drop(graph);
        drop(input);
        drop(output);
        assert_eq!(
            result::ctx::get_current().expect("query context after surviving child drops"),
            None,
            "surviving child cleanup must restore no-current state"
        );
        observer
            .make_current()
            .expect("rebind observer before memory sample");
        let (free, _) = observer.mem_info().expect("sample device memory");
        if cycle >= 8 {
            free_samples.push(free);
        }
    }

    let min_free = *free_samples.iter().min().expect("32 memory samples");
    let max_free = *free_samples.iter().max().expect("32 memory samples");
    let span = max_free - min_free;
    eprintln!(
        "CARD305_PTX_LIFECYCLE cycles=40 samples={} min_free={} max_free={} span_bytes={}",
        free_samples.len(),
        min_free,
        max_free,
        span
    );
    assert!(
        span <= 64 * 1024 * 1024,
        "post-drop device-memory span {span} exceeded 64 MiB"
    );
}

#[test]
fn native_buffer_contract_roundtrip_ptx() {
    let ctx = match PtxContext::new() {
        Ok(ctx) => ctx,
        Err(error) => {
            eprintln!("SKIP native_buffer_contract_roundtrip_ptx: {error}");
            return;
        }
    };

    let f32_buffer = ctx.upload_f32(&[1.0, 2.0, 3.0, 4.0]).expect("upload f32");
    ctx.update_f32(&f32_buffer, &[10.0, 20.0, 30.0, 40.0])
        .expect("update f32");
    assert_eq!(
        ctx.download_f32(&f32_buffer).expect("download f32"),
        [10.0, 20.0, 30.0, 40.0]
    );
    assert_eq!(f32_buffer.elem_count(), 4);
    assert_eq!(f32_buffer.byte_capacity(), 16);
    assert_eq!(f32_buffer.element(), BufferElement::F32);

    let i32_buffer = ctx.upload_i32(&[1, 2, 3, 4]).expect("upload i32");
    assert_eq!(
        ctx.download_i32(&i32_buffer).expect("download i32"),
        [1, 2, 3, 4]
    );
    assert_eq!(i32_buffer.elem_count(), 4);
    assert_eq!(i32_buffer.byte_capacity(), 16);
    assert_eq!(i32_buffer.element(), BufferElement::I32);

    let f16 = ctx
        .upload_f16(&[0x00, 0x3c, 0x00, 0xc0])
        .expect("upload f16");
    assert_eq!(f16.elem_count(), 2);
    assert_eq!(f16.byte_capacity(), 4);
    assert_eq!(f16.element(), BufferElement::F16);
    assert!(matches!(
        ctx.download_f32(&f16),
        Err(PtxError::RepresentationMismatch { .. })
    ));
    assert!(matches!(
        ctx.update_f32(&f16, &[1.0, -2.0]),
        Err(PtxError::RepresentationMismatch { .. })
    ));
    assert!(matches!(
        ctx.download_i32(&f32_buffer),
        Err(PtxError::RepresentationMismatch { .. })
    ));

    let clone = f16.clone();
    assert_eq!(clone.elem_count(), f16.elem_count());
    assert_eq!(clone.byte_capacity(), f16.byte_capacity());
    assert_eq!(clone.element(), f16.element());
}

// --- card 628: BinaryOpNoContract on real CUDA hardware (SC-001) -------------------------------------

/// `n` `(a, b, c, d)` rows where `a*b - c*d` computed with two separate f32 roundings differs from the
/// same expression with the second multiply and the subtract fused into one rounding step - i.e.
/// genuinely FMA-sensitive rows, found by a small deterministic search (splitmix64). Mirrors
/// `poot-runtime`'s and `poot-rocm-runtime`'s own `no_contract_fma_sensitive_rows` (kept crate-local
/// rather than shared, matching this crate's existing self-contained-fixture style).
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

/// Compile `poot_test_util::kernel_fixtures::no_contract_probe_kernel(marked)` for `Target::Nvptx` through the
/// real production toolchain (the same `poot_codegen::compile`/`kernel_handle` pair every other backend's
/// device test uses - `poot-runtime`'s `no_contract_tests`, `poot-rocm-runtime`'s `no_contract_*`).
fn no_contract_nvptx_kernel(marked: bool) -> poot_runtime_common::CompiledKernel {
    let tag = if marked { "marked" } else { "unmarked" };
    let body = poot_test_util::kernel_fixtures::no_contract_probe_kernel(marked);
    let dir = std::env::temp_dir().join("poot-ptx-runtime-no-contract-test");
    std::fs::create_dir_all(&dir).expect("create fixture artifact dir");
    let out = poot_codegen::artifact_path(
        &dir,
        &format!("no_contract_{tag}"),
        poot_codegen::Target::Nvptx,
    );
    poot_codegen::compile(&body, poot_codegen::Target::Nvptx, &out)
        .unwrap_or_else(|e| panic!("compile no_contract_{tag}: {e}"));
    let bytes =
        std::fs::read(&out).unwrap_or_else(|e| panic!("read compiled no_contract_{tag}: {e}"));
    poot_codegen::kernel_handle(&body, poot_codegen::Target::Nvptx, bytes)
}

/// SC-001 (dquant.md section 7 R1): the marked packed-dequant-style `a*b - c*d` kernel
/// dispatched on real CUDA hardware, checked against the unfused scalar reference bit for bit on
/// FMA-sensitive rows. This is real acceptance evidence that the marked kernel matches the CPU reference on
/// real hardware today - and it is not a lucky/undocumented default: `ptxas` is REQUIRED to honour it, per
/// PTX ISA 9.4 SS9.7.3.4 ("sub", Notes) - "a sub instruction with an explicit rounding modifier is treated
/// conservatively by the code optimizer" (contrasted with a modifier-less sub, which "may be optimized
/// aggressively... to use fused-multiply-add instructions"). `poot-codegen`'s own toolchain (`llc`, no
/// `ptxas`) always emits that explicit `.rn` for ordinary float sub, marked or not, so the marked and
/// unmarked kernels compile to byte-identical, ISA-protected `.ptx` text
/// (`poot-codegen/tests/no_contract.rs`'s `nvptx_marked_subtract_emits_constrained_intrinsic_under_strictfp`
/// proves this directly and asserts the `.rn` is present, no GPU needed) - which is also why the unmarked
/// kernel doesn't diverge on real NVPTX hardware either (batch #4, RTX 4090/driver 610.43.02,
/// 2026-09-29). There is deliberately no "unmarked diverges" counterpart test on Nvptx (removed 2026-09-29,
/// card 628 follow-up): the ISA guarantee above covers both kernels identically, so that assertion would
/// only ever restate it under a name that implies the opposite (that the unmarked kernel is unprotected),
/// which the same batch #4 run showed is false. Skips (passes) with no CUDA device; `POOT_REQUIRE_PTX=1`
/// turns that into a loud failure instead (`PtxContext::new`'s own `require_gpu_check`), for a pod run that
/// must not silently skip.
#[test]
fn no_contract_marker_matches_unfused_reference_on_fma_sensitive_rows() {
    let ctx = match PtxContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "SKIP no_contract_marker_matches_unfused_reference_on_fma_sensitive_rows (no PTX device): {e}"
            );
            return;
        }
    };
    let rows = no_contract_fma_sensitive_rows(64);
    let a: Vec<f32> = rows.iter().map(|r| r.0).collect();
    let b: Vec<f32> = rows.iter().map(|r| r.1).collect();
    let c: Vec<f32> = rows.iter().map(|r| r.2).collect();
    let d: Vec<f32> = rows.iter().map(|r| r.3).collect();
    let want: Vec<f32> = rows.iter().map(|&(a, b, c, d)| a * b - c * d).collect();

    let kernel = no_contract_nvptx_kernel(true);
    let a_buf = ctx.upload_f32(&a).expect("upload a");
    let b_buf = ctx.upload_f32(&b).expect("upload b");
    let c_buf = ctx.upload_f32(&c).expect("upload c");
    let d_buf = ctx.upload_f32(&d).expect("upload d");
    let out_buf = ctx.alloc_f32(rows.len()).expect("alloc out");
    let n = rows.len() as u32;
    ctx.dispatch_dev(
        "no_contract_probe",
        &kernel,
        [n, 1, 1],
        [n, 1, 1],
        &[&a_buf, &b_buf, &c_buf, &d_buf],
        &[
            a_buf.elem_count(),
            b_buf.elem_count(),
            c_buf.elem_count(),
            d_buf.elem_count(),
        ],
        &out_buf,
        out_buf.elem_count(),
    )
    .expect("dispatch no_contract_probe (marked)");
    let got = ctx.download_f32(&out_buf).expect("download out");
    assert_eq!(
        got, want,
        "the NoContraction-marked NVPTX kernel diverged from the unfused scalar reference on an \
         FMA-sensitive row - ptxas contracted the marked subtract with a producing multiply despite the \
         constrained intrinsic"
    );
}

/// Card 549 SC-005 (R471-010): a `PtxBuffer` (and the `PtxContext` that allocated it) are `Send` - they
/// can be built on one thread and moved, whole, into another, which is exactly the shape
/// `poot-serve`'s model-load-thread-to-engine-loop-thread handoff needs (`crate` module doc). This only
/// compiles because every owning handle's shared native state (`PtxOwner`) is `Arc`-backed, not
/// `Rc`-backed: `Rc<T>` is never `Send` regardless of what it owns, so this function would not even
/// type-check against the pre-549 `Rc`-based buffer.
///
/// Mutation (R471-010): revert `PtxBuffer`'s `alloc: Arc<PtxAlloc>` field (and `PtxAlloc::owner`) back to
/// `Rc`; this test stops compiling with `` `Rc<PtxAlloc>` cannot be sent between threads safely `` (E0277),
/// which is the row's required red signal (a deletion/soundness row is proven by the compiler, ADR-0090).
#[test]
fn ptx_buffer_and_its_context_move_into_another_thread() {
    let Ok(ctx) = PtxContext::new() else {
        eprintln!("SKIP ptx_buffer_and_its_context_move_into_another_thread: no NVIDIA GPU");
        return;
    };
    let buffer = ctx.upload_f32(&[1.0, 2.0, 3.0, 4.0]).expect("upload f32");
    let handle = std::thread::spawn(move || {
        // A moved `PtxContext` is current only on the thread that built it (module doc); establish it
        // here before touching either handle.
        ctx.make_current().expect("make_current on the new thread");
        ctx.download_f32(&buffer)
            .expect("download f32 on the new thread")
    });
    assert_eq!(
        handle.join().expect("worker thread panicked"),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}
