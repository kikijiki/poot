//! The engine over an NPU device: a region program loads its regions through the device's `RegionLoader` and
//! dispatches each as one finite invocation (POOT-1117, POOT-1164, POOT-1244). The devices here are recorders
//! with an NPU target; there is no NPU in this process, so no row claims execution (the device lanes' rows do).
//! Model-free.

use poot_graph_plan::{
    CompileOptions, multi_device::topology::DeviceId, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::TargetSet, staged::compile_staged,
};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::BufferRole;
use poot_target::Queried;
use poot_weights::{AttnRole, FfnRole, WeightId, WeightMap, WeightRole, WeightView};

mod aie_engine {
    use std::num::NonZeroU32;
    use std::sync::Arc;

    use super::*;
    use poot_codegen::{KernelCache, Target as CodegenTarget};
    use poot_executor::{
        CopyMechanism, Device, DeviceTime, Dispatch, Engine, EntryId, ExecError, ExecutableId,
        Executor, LaunchKind, LoadError, NoSync, RegionLoader, StepInputs,
    };
    use poot_graph_ir::Builder;
    use poot_graph_ir::TensorType;
    use poot_graph_plan::{NpuTarget, Target};
    use poot_npu::test_support::scripted_toolchain;
    use poot_npu::{AieLoadError, AieRegionLoader, npu_lowering};
    use poot_npu_artifacts::AieInvocation;
    use poot_target::{AieCaps, AieTargetId, BufferStorage, TargetId};
    use poot_tensor::DType;

    const A: WeightId = WeightId::layer(0, WeightRole::Attn(AttnRole::Q));
    const B: WeightId = WeightId::layer(0, WeightRole::Ffn(FfnRole::Gate));
    const C: WeightId = WeightId::layer(0, WeightRole::Ffn(FfnRole::Up));

    /// What the device was handed: every loaded kernel's target, binding count and instruction words, and
    /// every dispatch's launch and argument element counts.
    #[derive(Default)]
    struct Seen {
        loaded: Vec<(TargetId, usize, usize)>,
        dispatches: Vec<(LaunchKind, Vec<usize>, u32)>,
    }

    struct AieRecorder {
        caps: AieCaps,
        loader: AieRegionLoader,
        seen: Seen,
        buffers: usize,
        /// The role of every buffer the engine allocated, in order.
        roles: Vec<BufferRole>,
    }

    impl Device for AieRecorder {
        type Buffer = usize;
        type Kernel = usize;
        type Recording = ();
        type Error = std::convert::Infallible;

        fn copy_mechanism(&mut self) -> CopyMechanism<'_, std::convert::Infallible> {
            CopyMechanism::Native
        }

        fn backend_name(&self) -> &'static str {
            "fake"
        }

        fn target(&self) -> Target {
            Target::Npu(NpuTarget::Aie {
                target: AieTargetId::Xdna2,
                caps: self.caps,
            })
        }

        fn region_loader(&self) -> Option<&dyn RegionLoader> {
            Some(&self.loader)
        }
        fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
            Vec::new()
        }
        fn allocate(
            &mut self,
            role: BufferRole,
            _storage: BufferStorage,
            _elems: usize,
        ) -> Result<usize, Self::Error> {
            self.roles.push(role);
            self.buffers += 1;
            Ok(self.buffers - 1)
        }
        fn buffer_elems(&self, _buffer: &usize) -> usize {
            // This fake keeps no storage, so it states no bound: the length check passes every dispatch.
            usize::MAX
        }
        fn load_kernel(
            &mut self,
            _key: &str,
            kernel: poot_codegen::CompiledKernel,
        ) -> Result<usize, Self::Error> {
            let poot_codegen::KernelCode::Aie(artifact) = kernel.code() else {
                panic!("an AIE device was handed a byte artifact");
            };
            self.seen.loaded.push((
                kernel.target(),
                kernel.args().len(),
                artifact.instructions().len() / 4,
            ));
            Ok(self.seen.loaded.len() - 1)
        }
        fn begin(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), Self::Error> {
            let mut elems: Vec<usize> = d.inputs.iter().map(|arg| arg.elems).collect();
            elems.push(d.output.elems);
            self.seen.dispatches.push((d.launch, elems, d.work as u32));
            Ok(())
        }
        fn copy(&mut self, _src: &usize, _dst: &usize) -> Result<(), Self::Error> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn replay(&mut self, _recording: &()) -> Result<(), Self::Error> {
            Ok(())
        }
        fn write_with(
            &mut self,
            _dst: &usize,
            _offset: usize,
            _len: usize,
            _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
        ) -> Result<(), Self::Error> {
            Ok(())
        }
        fn read(&mut self, _src: &usize, _out: &mut [u8]) -> Result<(), Self::Error> {
            Ok(())
        }
        fn synchronize(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn device_time(&self) -> DeviceTime {
            DeviceTime::Unknown
        }
        fn abort(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    fn caps() -> AieCaps {
        // SYNTHETIC limits, roomy for a `[2, 3]` add: not a device measurement.
        AieCaps {
            target: AieTargetId::Xdna2,
            tile_local_bytes: Queried::Known(1 << 16),
            host_bindings: Queried::Known(8),
            instruction_bytes: Queried::Known(1 << 20),
            memory_budget_bytes: Queried::Known(1 << 30),
            region: poot_target::aie::RegionLimits {
                compute_tiles: Queried::Known(32),
                tile_input_channels: Queried::Known(2),
                tile_output_channels: Queried::Known(2),
                host_input_streams: Queried::Known(16),
                host_output_streams: Queried::Known(16),
            },
        }
    }

    /// `a + b` over two F32 `[2, 3]` weights, compiled for the recorder's AIE target and added as an entry.
    fn load_add(engine: &mut Engine<AieRecorder>) -> Result<(ExecutableId, EntryId), ExecError> {
        load_graph(engine, &[A, B], |b, w| {
            b.binary(poot_graph_ir::BinOp::Add, w[0], w[1]).unwrap()
        })
    }

    /// The graph `build` makes over F32 `[2, 3]` weights named by `ids`, compiled for the recorder's AIE
    /// target and added as an entry.
    fn load_graph(
        engine: &mut Engine<AieRecorder>,
        ids: &[WeightId],
        build: impl FnOnce(&Builder, &[poot_graph_ir::Traced]) -> poot_graph_ir::Traced,
    ) -> Result<(ExecutableId, EntryId), ExecError> {
        let mut store = WeightStore::builder();
        for &id in ids {
            let bytes: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
            store
                .insert(
                    id.const_name(),
                    WeightEntry::Dense(
                        DenseWeight::try_new(DType::F32, vec![2, 3], bytes.into()).unwrap(),
                    ),
                )
                .unwrap();
        }
        let store = Arc::new(store.build());
        let b = Builder::for_tests();
        let consts: Vec<_> = ids
            .iter()
            .map(|id| {
                b.constant(&id.const_name(), TensorType::f32(vec![2, 3]))
                    .unwrap()
            })
            .collect();
        let out = build(&b, &consts);
        let g = b.finish(out).unwrap();
        let mut weights = WeightMap::builder(&store);
        for &id in ids {
            weights
                .map(id, WeightView::Stored(id.const_name().into()))
                .unwrap();
        }
        let weights = weights.build();
        let program = compile_staged(
            &g,
            &TargetSet::single(DeviceId(0), engine.device().target()),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions {
                npu: Some(npu_lowering()),
                ..CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD)
            },
        )
        .expect("the add compiles for an AIE target");
        let exe = engine.load_weights(store, Arc::new(weights))?;
        let entry = engine.add_entry(exe, &program)?;
        Ok((exe, entry))
    }

    fn engine(caps: AieCaps, with_iron: bool, dir: &std::path::Path) -> Engine<AieRecorder> {
        Engine::new(AieRecorder {
            caps,
            loader: AieRegionLoader::with_toolchain(
                caps,
                scripted_toolchain(dir, with_iron),
                KernelCache::at(dir.join("cache"), CodegenTarget::AieCore),
            ),
            seen: Seen::default(),
            buffers: 0,
            roles: Vec::new(),
        })
    }

    /// The loader's own typed cause behind a refused region. The executor seam boxes the cause
    /// (`RegionLoadError`), so the executor cannot name the vendor's variants; this downcast is where the
    /// typed cause is asserted, and it panics when a loader boxes anything else.
    fn region_cause(error: &LoadError) -> &AieLoadError {
        match error {
            LoadError::Region { source, .. } => source
                .downcast_ref::<AieLoadError>()
                .unwrap_or_else(|| panic!("not an AMD loader cause: {source}")),
            other => panic!("expected a region refusal, got {other:?}"),
        }
    }

    fn load_error(error: ExecError) -> LoadError {
        match error {
            ExecError::Load(load) => *load,
            other => panic!("expected a load error, got {other:?}"),
        }
    }

    /// An AIE device's add is linked through `compile_aie` against the plan's own ABI, loaded as one linked
    /// artifact of three bindings and the stream's one instruction word, and dispatched as one finite
    /// `LaunchKind::Aie` invocation naming that stream; the three arguments carry the planned element counts.
    /// Mutation: have the engine's AIE arm hand the device `LaunchKind::Grid { threads: launch.grid, .. }`
    /// (the pre-POOT-1117 grid carrier); the launch is no `LaunchKind::Aie` and this row goes red.
    #[test]
    fn an_aie_device_loads_a_linked_artifact_and_dispatches_one_aie_invocation() {
        let dir = poot_test_util::unique_temp_path("aie-engine");
        std::fs::create_dir_all(&*dir).unwrap();
        let mut engine = engine(caps(), true, &dir);
        let (exe, entry) = load_add(&mut engine).expect("the entry loads through compile_aie");
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .expect("the recorded step replays");
        let seen = &engine.device().seen;
        assert_eq!(
            seen.loaded,
            vec![(TargetId::Aie(AieTargetId::Xdna2), 3, 1)],
            "one kernel: three bindings, one instruction word"
        );
        assert_eq!(seen.dispatches.len(), 1);
        let (launch, elems, _) = &seen.dispatches[0];
        assert_eq!(
            *launch,
            LaunchKind::Aie(AieInvocation::new(
                AieTargetId::Xdna2,
                NonZeroU32::new(1).unwrap()
            ))
        );
        assert_eq!(
            elems,
            &vec![6, 6, 6],
            "two reads and the write, six elements each"
        );
    }

    /// A chain of three equations is one region: the engine loads one linked artifact whose host side is the
    /// three weights read and the one value returned, and dispatches it once. The two values in between exist
    /// only inside the artifact: the engine allocates a buffer for the returned value and for no other
    /// computed value.
    /// Mutation: have `package` close a region after every equation (`fits` returns an error for any
    /// region of two); three kernels are loaded and three dispatched, and this row goes red.
    #[test]
    fn a_chain_is_one_linked_region_dispatched_once_with_no_buffer_for_its_intermediates() {
        let dir = poot_test_util::unique_temp_path("aie-engine-chain");
        std::fs::create_dir_all(&*dir).unwrap();
        let mut engine = engine(caps(), true, &dir);
        let (exe, entry) = load_graph(&mut engine, &[A, B, C], |b, w| {
            let first = b.binary(poot_graph_ir::BinOp::Add, w[0], w[1]).unwrap();
            let second = b.binary(poot_graph_ir::BinOp::Mul, first, w[2]).unwrap();
            b.binary(poot_graph_ir::BinOp::Add, second, w[0]).unwrap()
        })
        .expect("the chain loads through compile_aie");
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .expect("the recorded step replays");
        let seen = &engine.device().seen;
        assert_eq!(
            seen.loaded,
            vec![(TargetId::Aie(AieTargetId::Xdna2), 4, 1)],
            "one artifact for the three equations: three reads and the returned value"
        );
        assert_eq!(seen.dispatches.len(), 1, "one invocation for the region");
        assert_eq!(seen.dispatches[0].1, vec![6, 6, 6, 6]);
        let activations = engine
            .device()
            .roles
            .iter()
            .filter(|role| **role == BufferRole::Activation)
            .count();
        assert_eq!(
            activations, 1,
            "only the returned value has an arena slot; the two values in between have none"
        );
    }

    /// What only the linked artifact can answer is checked after linking and before the artifact is wrapped:
    /// this target's instruction limit is below the stream the driver wrote. The engine refuses it as the
    /// typed `a region refusal` and the device is never handed a kernel. Mutation: drop the
    /// `validate_linked_artifact` call in `AieRegionLoader::compile`; the kernel loads and this row goes red.
    #[test]
    fn a_linked_artifact_past_the_targets_instruction_limit_is_refused_before_load() {
        let dir = poot_test_util::unique_temp_path("aie-engine-instr");
        std::fs::create_dir_all(&*dir).unwrap();
        let tight = AieCaps {
            instruction_bytes: poot_target::Queried::Known(3),
            ..caps()
        };
        let mut engine = engine(tight, true, &dir);
        let error = load_error(load_add(&mut engine).unwrap_err());
        assert!(
            matches!(
                region_cause(&error),
                AieLoadError::Limit(poot_npu_artifacts::AieLimitRefusal::Exceeded {
                    resource: "instruction bytes",
                    needed: 4,
                    available: 3,
                })
            ),
            "{error:?}"
        );
        assert!(engine.device().seen.loaded.is_empty(), "nothing was loaded");
    }

    /// A missing IRON driver is the compiler's typed `ToolchainMissing`, carried as the engine's
    /// `LoadError::Region`; nothing is loaded. Mutation: swallow a failed `compile_aie` and claim a
    /// loaded kernel index anyway; `add_entry` succeeds and this row goes red.
    #[test]
    fn a_missing_link_driver_is_a_typed_load_error_and_loads_nothing() {
        let dir = poot_test_util::unique_temp_path("aie-engine-no-iron");
        std::fs::create_dir_all(&*dir).unwrap();
        let mut engine = engine(caps(), false, &dir);
        let error = load_error(load_add(&mut engine).unwrap_err());
        assert!(
            matches!(
                region_cause(&error),
                AieLoadError::Compile(poot_npu::AieCompileError::ToolchainMissing {
                    stage: poot_npu::AieStage::Iron,
                    ..
                })
            ),
            "{error:?}"
        );
        assert!(engine.device().seen.loaded.is_empty(), "nothing was loaded");
    }
}

/// POOT-1164: the engine's Intel NPU arm. A device whose target is an Intel NPU loads one graph artifact per
/// program (the OpenVINO IR the planner's region emits, with its argument ABI), and every step dispatches it
/// once as a `LaunchKind::VpuGraph` invocation of the artifact it loaded. The device here is a recorder with an
/// Intel NPU target; there is no NPU in this process, so no row here claims execution (the device lane's rows
/// do).
mod vpu_engine {
    use std::sync::Arc;

    use super::*;
    use poot_codegen::KernelCode;
    use poot_executor::{
        CopyMechanism, Device, DeviceTime, Dispatch, Engine, EntryId, ExecError, ExecutableId,
        Executor, LaunchKind, NoSync, RegionLoader, StepInputs,
    };
    use poot_graph_ir::Builder;
    use poot_graph_ir::TensorType;
    use poot_graph_plan::{NpuTarget, Target};
    use poot_npu::{VpuRegionLoader, npu_lowering};
    use poot_npu_artifacts::VpuInvocation;
    use poot_target::{BufferStorage, TargetId, VpuTargetId};
    use poot_target::{VendorCompilerId, VpuCaps};
    use poot_tensor::DType;

    const A: WeightId = WeightId::layer(0, WeightRole::Attn(AttnRole::Q));
    const B: WeightId = WeightId::layer(0, WeightRole::Ffn(FfnRole::Gate));
    const C: WeightId = WeightId::layer(0, WeightRole::Ffn(FfnRole::Up));

    /// What the device was handed: every loaded artifact's digest, input and output counts, and every
    /// dispatch's launch and argument element counts.
    #[derive(Default)]
    struct Seen {
        loaded: Vec<(u64, usize, usize)>,
        dispatches: Vec<(LaunchKind, Vec<usize>)>,
    }

    struct VpuRecorder {
        loader: VpuRegionLoader,
        seen: Seen,
        buffers: usize,
        roles: Vec<BufferRole>,
    }

    impl Device for VpuRecorder {
        type Buffer = usize;
        type Kernel = usize;
        type Recording = ();
        type Error = std::convert::Infallible;

        fn copy_mechanism(&mut self) -> CopyMechanism<'_, std::convert::Infallible> {
            CopyMechanism::Native
        }

        fn backend_name(&self) -> &'static str {
            "fake"
        }

        fn target(&self) -> Target {
            Target::Npu(NpuTarget::Vpu {
                target: VpuTargetId::Npu4,
                // SYNTHETIC: an NPU4 with a made-up compiler version, not a device measurement.
                caps: VpuCaps {
                    target: VpuTargetId::Npu4,
                    compiler: Queried::Known(VendorCompilerId { major: 8, minor: 3 }),
                    memory_budget_bytes: Queried::Unknown,
                    narrowing: None,
                },
            })
        }

        fn region_loader(&self) -> Option<&dyn RegionLoader> {
            Some(&self.loader)
        }
        fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
            Vec::new()
        }
        fn allocate(
            &mut self,
            role: BufferRole,
            _storage: BufferStorage,
            _elems: usize,
        ) -> Result<usize, Self::Error> {
            self.roles.push(role);
            self.buffers += 1;
            Ok(self.buffers - 1)
        }
        fn buffer_elems(&self, _buffer: &usize) -> usize {
            // This fake keeps no storage, so it states no bound: the length check passes every dispatch.
            usize::MAX
        }
        fn load_kernel(
            &mut self,
            _key: &str,
            kernel: poot_codegen::CompiledKernel,
        ) -> Result<usize, Self::Error> {
            let KernelCode::Vpu(artifact) = kernel.code() else {
                panic!("an Intel NPU device was handed a non-graph artifact");
            };
            assert_eq!(kernel.target(), TargetId::Vpu(VpuTargetId::Npu4));
            self.seen.loaded.push((
                artifact.digest(),
                artifact.inputs().len(),
                artifact.outputs().len(),
            ));
            Ok(self.seen.loaded.len() - 1)
        }
        fn begin(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn dispatch(&mut self, d: Dispatch<'_, Self>) -> Result<(), Self::Error> {
            let mut elems: Vec<usize> = d.inputs.iter().map(|arg| arg.elems).collect();
            elems.push(d.output.elems);
            self.seen.dispatches.push((d.launch, elems));
            Ok(())
        }
        fn copy(&mut self, _src: &usize, _dst: &usize) -> Result<(), Self::Error> {
            Ok(())
        }
        fn finish(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn replay(&mut self, _recording: &()) -> Result<(), Self::Error> {
            Ok(())
        }
        fn write_with(
            &mut self,
            _dst: &usize,
            _offset: usize,
            _len: usize,
            _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
        ) -> Result<(), Self::Error> {
            Ok(())
        }
        fn read(&mut self, _src: &usize, _out: &mut [u8]) -> Result<(), Self::Error> {
            Ok(())
        }
        fn synchronize(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn device_time(&self) -> DeviceTime {
            DeviceTime::Unknown
        }
        fn abort(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    /// The graph `build` makes over F16 `[2, 3]` weights named by `ids`, compiled for the recorder's Intel NPU
    /// target and added as an entry.
    fn load_graph(
        engine: &mut Engine<VpuRecorder>,
        ids: &[WeightId],
        build: impl FnOnce(&Builder, &[poot_graph_ir::Traced]) -> poot_graph_ir::Traced,
    ) -> Result<(ExecutableId, EntryId), ExecError> {
        let mut store = WeightStore::builder();
        for &id in ids {
            let bytes: Vec<u8> = (0..6u16).flat_map(|i| (0x3c00 + i).to_le_bytes()).collect();
            store
                .insert(
                    id.const_name(),
                    WeightEntry::Dense(
                        DenseWeight::try_new(DType::F16, vec![2, 3], bytes.into()).unwrap(),
                    ),
                )
                .unwrap();
        }
        let store = Arc::new(store.build());
        let b = Builder::for_tests();
        let consts: Vec<_> = ids
            .iter()
            .map(|id| {
                b.constant(&id.const_name(), TensorType::new(vec![2, 3], DType::F16))
                    .unwrap()
            })
            .collect();
        let out = build(&b, &consts);
        let g = b.finish(out).unwrap();
        let mut weights = WeightMap::builder(&store);
        for &id in ids {
            weights
                .map(id, WeightView::Stored(id.const_name().into()))
                .unwrap();
        }
        let weights = weights.build();
        let program = compile_staged(
            &g,
            &TargetSet::single(DeviceId(0), engine.device().target()),
            &Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions {
                npu: Some(npu_lowering()),
                ..CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD)
            },
        )
        .expect("the graph compiles for an Intel NPU target");
        let exe = engine.load_weights(store, Arc::new(weights))?;
        let entry = engine.add_entry(exe, &program)?;
        Ok((exe, entry))
    }

    fn engine() -> Engine<VpuRecorder> {
        // `Engine::new` over an Intel NPU device must not need a kernel-body compiler.
        Engine::new(VpuRecorder {
            loader: VpuRegionLoader::new(VpuTargetId::Npu4),
            seen: Seen::default(),
            buffers: 0,
            roles: Vec::new(),
        })
    }

    /// A chain of three equations over three weights is one graph artifact whose host side is the three
    /// weights read and the one value returned, dispatched once as a graph invocation of that same artifact.
    /// The two values in between exist only inside the graph: the engine allocates a buffer for the returned
    /// value and for no other computed value.
    ///
    /// Mutation: have `package` return no region for `ArtifactGranularity::Graph`; the load is refused ("a
    /// graph node outside any region") and nothing is loaded.
    #[test]
    fn a_chain_is_one_graph_artifact_dispatched_once_with_no_buffer_for_its_intermediates() {
        let mut engine = engine();
        let (exe, entry) = load_graph(&mut engine, &[A, B, C], |b, w| {
            let first = b.binary(poot_graph_ir::BinOp::Add, w[0], w[1]).unwrap();
            let second = b.binary(poot_graph_ir::BinOp::Mul, first, w[2]).unwrap();
            b.binary(poot_graph_ir::BinOp::Add, second, w[0]).unwrap()
        })
        .expect("the chain loads");
        engine
            .step(exe, entry, &StepInputs::new(), &mut NoSync)
            .expect("the recorded step replays");
        let seen = &engine.device().seen;
        assert_eq!(seen.loaded.len(), 1, "one artifact for the three equations");
        let (digest, inputs, outputs) = seen.loaded[0];
        assert_eq!(
            (inputs, outputs),
            (3, 1),
            "three weights read, one value returned"
        );
        assert_eq!(seen.dispatches.len(), 1, "one invocation for the graph");
        assert_eq!(
            seen.dispatches[0].0,
            LaunchKind::VpuGraph(VpuInvocation::new(VpuTargetId::Npu4, digest)),
            "the invocation names the artifact the engine loaded"
        );
        assert_eq!(seen.dispatches[0].1, vec![6, 6, 6, 6]);
        let activations = engine
            .device()
            .roles
            .iter()
            .filter(|role| **role == BufferRole::Activation)
            .count();
        assert_eq!(
            activations, 1,
            "only the returned value has an arena slot; the two values in between have none"
        );
    }

    /// Two entries over the same graph load the artifact once: the engine keys it by its digest.
    ///
    /// Mutation: drop the `region_kernels` lookup in `Engine::region_kernel`; the device loads the artifact twice.
    #[test]
    fn the_same_graph_added_twice_is_loaded_once() {
        let mut engine = engine();
        for _ in 0..2 {
            load_graph(&mut engine, &[A, B], |b, w| {
                b.binary(poot_graph_ir::BinOp::Add, w[0], w[1]).unwrap()
            })
            .expect("the add loads");
        }
        assert_eq!(engine.device().seen.loaded.len(), 1);
    }
}
