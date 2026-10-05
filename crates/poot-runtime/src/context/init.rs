use crate::*;

impl Context {
    /// A context with device-timing off (the default; no per-dispatch overhead).
    pub fn new() -> Result<Self, RuntimeError> {
        pollster::block_on(Self::new_async(false, 0)).map_err(require_gpu_check)
    }

    /// A context for the typed Card 552 timing path. `max_in_flight_queries` is the declared bound
    /// (review F3) on concurrently un-drained timestamp-readback resources - `0` means detailed
    /// collection is off entirely (the counters-only default: no query resources are ever created,
    /// and [`Context::drain_device_timing`] always reports nothing); any positive value turns
    /// collection on and bounds it.
    pub fn new_with_device_timing(max_in_flight_queries: usize) -> Result<Self, RuntimeError> {
        pollster::block_on(Self::new_async(
            max_in_flight_queries > 0,
            max_in_flight_queries,
        ))
        .map_err(require_gpu_check)
    }

    /// Whether this context was built with [`Context::new_with_device_timing`] at a positive bound
    /// (Card 552): read by [`Context::submit_encoded`] (the one submit path, review F1) to decide
    /// whether to request per-dispatch timestamps on the replayed work at all.
    pub fn device_timing(&self) -> bool {
        self.device_timing.get()
    }

    /// Card 552: discard every pending/already-drained device-timing readback without
    /// reading it, and reset the next-dispatch-index counter - called once at the start of every
    /// `WgpuDevice::replay` (and, defensively, from `abort`), so a step that failed before
    /// `drain_device_timing` ever ran cannot contaminate the next step's sum/span/kind-name
    /// buckets.
    pub fn discard_device_timing(&self) {
        self.pending_device_timing.borrow_mut().clear();
        self.drained_device_timing.take();
        self.next_dispatch_index.set(0);
    }

    /// Whether device-time recording is available (the adapter supports `TIMESTAMP_QUERY`). Test-only: its
    /// one caller is `tests::profiled_dispatch_tests`.
    #[cfg(test)]
    pub(crate) fn timestamps_available(&self) -> bool {
        self.timestamps
    }

    /// The device `max_compute_workgroups_per_dimension` used to fold oversized 1-D dispatch grids.
    pub fn max_workgroups(&self) -> u32 {
        self.max_workgroups
    }

    /// The raw device `max_buffer_size` from wgpu `Limits` (bytes): the largest single buffer
    /// `Device::create_buffer` accepts. Queried at init, never hardcoded (Card 453 D1).
    pub fn max_buffer_size(&self) -> u64 {
        self.max_buffer_size
    }

    /// The **effective** single-buffer limit for allocation planning: `min` of the raw device
    /// `max_buffer_size` and [`AMD_RADV_SAFE_MAX_BUFFER_BYTES`] on AMD Vulkan (RADV) adapters,
    /// else the raw limit. This is the number exact-model traces must consume (Card 453 D1);
    /// see the constant's doc for the observed RADV gfx-ring wedge at the raw device max.
    pub fn effective_max_buffer_size(&self) -> u64 {
        self.effective_max_buffer_size
    }

    /// The device-capability descriptor the planner reads (card 522): the buffer, LDS and grid limits
    /// and the tensor-core family are this device's own query; the display-watchdog budget and the
    /// tiled-GEMM miscompile ceiling are not device-queryable at all (they are calibrated from observed
    /// hangs and wrong output, not reported by any driver API), so they are the vendor-scoped default for
    /// AMD Vulkan (RADV) - [`poot_target::DeviceCaps::wgpu_rdna3_igpu`]'s values - on `self.is_amd_radv`, else absent.
    pub fn device_caps(&self) -> poot_target::DeviceCaps {
        let (watchdog_budget, known_miscompiles) = if self.is_amd_radv {
            let d = poot_target::DeviceCaps::wgpu_rdna3_igpu();
            (d.watchdog_budget, d.known_miscompiles)
        } else {
            (None, poot_target::KnownMiscompiles::default())
        };
        let launch = LaunchCaps::from_wgpu(
            &self.device.limits(),
            self._adapter.features().contains(wgpu::Features::SUBGROUP),
            {
                let info = self._adapter.get_info();
                (info.subgroup_min_size, info.subgroup_max_size)
            },
        );
        poot_target::DeviceCaps {
            max_buffer_bytes: self.effective_max_buffer_size,
            lds_bytes: self.lds_bytes,
            max_workgroup_size: launch.max_workgroup_size,
            max_workgroup_invocations: launch.max_workgroup_invocations,
            subgroup: launch.subgroup,
            // No driver API reports a safe per-dispatch work bound.
            max_dispatch_work: poot_target::Queried::Unknown,
            max_grid: [self.max_workgroups; 3],
            watchdog_budget,
            tensor_core: self.tensor_core,
            known_miscompiles,
            compute_units: self.compute_units,
        }
    }

    pub(crate) async fn new_async(
        device_timing: bool,
        max_in_flight_queries: usize,
    ) -> Result<Self, RuntimeError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| RuntimeError::NoAdapter(e.to_string()))?;

        // PASSTHROUGH_SHADERS is required for raw SPIR-V; the rest are enabled only if available.
        let mut features = wgpu::Features::PASSTHROUGH_SHADERS;
        for opt in [
            wgpu::Features::SHADER_F64,
            wgpu::Features::SHADER_FLOAT32_ATOMIC,
            wgpu::Features::SHADER_F16,
            wgpu::Features::SUBGROUP,
        ] {
            if adapter.features().contains(opt) {
                features |= opt;
            }
        }
        // Device-time timing needs TIMESTAMP_QUERY; request it only when the Card 552 typed
        // device-timing path is on, and degrade (host wall + bytes only, or `DeviceTime::Unknown`) if
        // the adapter lacks it (FR-005).
        let timestamps =
            device_timing && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        if timestamps {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        // Raise the conservative downlevel limits to the adapter maxima (storage buffers per stage,
        // workgroup size, buffer/binding sizes) so multi-slot and large-weight kernels fit. Also take the
        // adapter's per-dimension workgroup-count cap: dispatch folding uses the device limit rather
        // than a hardcoded 65535.
        let a = adapter.limits();
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = a.max_storage_buffers_per_shader_stage;
        limits.max_compute_invocations_per_workgroup = a.max_compute_invocations_per_workgroup;
        limits.max_compute_workgroup_size_x = a.max_compute_workgroup_size_x;
        limits.max_buffer_size = a.max_buffer_size;
        limits.max_storage_buffer_binding_size = a.max_storage_buffer_binding_size;
        limits.max_compute_workgroups_per_dimension = a.max_compute_workgroups_per_dimension;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("poot-runtime"),
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| RuntimeError::DeviceInit(e.to_string()))?;
        // The granted device limit (source of truth for dispatch folding), not a hardcoded 65535.
        let max_workgroups = device.limits().max_compute_workgroups_per_dimension;
        // Card 453 D1: raw largest single buffer this device accepts (queried, not hardcoded).
        let max_buffer_size = device.limits().max_buffer_size;
        // Card 453 D1: effective limit for allocation planning. Scoped to AMD Vulkan (RADV,
        // vendor 0x1002) where a chunk at the raw device max wedged the gfx ring on first
        // dispatch; other vendors use the raw limit unchanged.
        let adapter_info = adapter.get_info();
        let is_amd_radv =
            adapter_info.backend == wgpu::Backend::Vulkan && adapter_info.vendor == 0x1002;
        let effective_max_buffer_size = if is_amd_radv {
            max_buffer_size.min(crate::AMD_RADV_SAFE_MAX_BUFFER_BYTES)
        } else {
            max_buffer_size
        };
        // Card 522: the LDS/shared-memory budget one dispatch may use. The granted device limit
        // (`device.limits()`, like `max_workgroups`/`max_buffer_size` above), not the adapter's
        // unrequested maximum: this crate's `required_limits` never raises
        // `max_compute_workgroup_storage_size` past `downlevel_defaults()`, so the adapter's own higher
        // ceiling (if any) is not actually available to a dispatch on this device.
        let lds_bytes = device.limits().max_compute_workgroup_storage_size;
        // Card 522 review: measured from `VK_KHR_cooperative_matrix` (via wgpu's own
        // `Adapter::cooperative_matrix_properties`, populated at adapter-capability-probe time, no
        // feature/extension request needed to read it), never parsed from the adapter's device-name
        // string - Mesa RADV reports a marketing codename ("AMD Radeon 8060S Graphics (RADV
        // STRIX_HALO)"), not a gfx code, so a name-based classifier would wrongly call this device's
        // real RDNA3 WMMA hardware absent. Every reported config is folded (a device can report more
        // than one shape); `wgpu::Backend::Vulkan` is the only backend this crate confirms the property
        // list is meaningfully populated for, so any other backend says `UnknownNotExposedByApi`.
        let tensor_core = if adapter_info.backend == wgpu::Backend::Vulkan {
            adapter
                .cooperative_matrix_properties()
                .into_iter()
                .map(|c| {
                    poot_target::TensorCoreSupport::from_cooperative_matrix_config(
                        c.m_size,
                        c.n_size,
                        c.k_size,
                        c.ab_type == wgpu::CooperativeScalarType::F16,
                        c.cr_type == wgpu::CooperativeScalarType::F32,
                    )
                })
                .fold(poot_target::TensorCoreSupport::None, |acc, x| {
                    acc.most_specific(x)
                })
        } else {
            poot_target::TensorCoreSupport::UnknownNotExposedByApi
        };
        // Card 653: wgpu has no compute-unit query, so read AMD's own Vulkan
        // shader-core properties through the wgpu-hal adapter where the driver exposes them; any other
        // device plans against the RDNA3 default, the one device family this backend's launch policy
        // is measured on.
        let compute_units = probe_amd_compute_units(&adapter)
            .unwrap_or(poot_target::DeviceCaps::wgpu_rdna3_igpu().compute_units);
        // Card 158 diagnostic (debug-only, off by default): a deferred wgpu validation error (e.g. one
        // surfacing only on `Device::poll`, like the real-35B batched-prefill panic) shows as a bare
        // "Validation Error" if the failing resource has no debug label. This handler covers the other
        // class (submission/creation-time errors routed through wgpu's error sink, which does consult this
        // callback, unlike the `Device::poll` fatal path that panics in `wgpu-core`'s formatter): it prints
        // the `Display` and `Debug` forms plus each link of the `source()` chain to stderr, then re-raises
        // (card 537, SC-003). Gated behind `POOT_WGPU_DEBUG` (one `env::var_os` read at context
        // creation): wgpu's own default uncaptured-error handler panics, so installing a custom one here
        // must never turn that into silent continuation - `POOT_WGPU_DEBUG` only adds detail to a fatal
        // error, it never decides whether one is fatal. Debugging-only per the Scope exception in exactly
        // this sense: the extra diagnostics never change whether or what panics, only what gets printed
        // first.
        if std::env::var_os("POOT_WGPU_DEBUG").is_some() {
            device.on_uncaptured_error(std::sync::Arc::new(|err: wgpu::Error| {
                eprintln!("[poot-wgpu-debug] uncaptured wgpu error (Display):\n{err}");
                eprintln!("[poot-wgpu-debug] uncaptured wgpu error (Debug):\n{err:?}");
                let mut source = std::error::Error::source(&err);
                let mut depth = 0usize;
                while let Some(s) = source {
                    eprintln!("[poot-wgpu-debug]   caused by [{depth}]: {s}");
                    source = s.source();
                    depth += 1;
                }
                panic!("[poot-wgpu-debug] uncaptured wgpu error (see diagnostics above): {err}");
            }));
        }
        Ok(Context {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            timestamps,
            device_timing: Cell::new(device_timing),
            max_in_flight_queries: Cell::new(max_in_flight_queries),
            drained_device_timing: RefCell::new(DeviceTimingAccumulator::default()),
            next_dispatch_index: Cell::new(0),
            pending_device_timing: RefCell::new(Vec::new()),
            max_workgroups,
            max_buffer_size,
            effective_max_buffer_size,
            lds_bytes,
            tensor_core,
            compute_units,
            is_amd_radv,
            pipelines: RefCell::new(HashMap::new()),
            pipeline_builds: Default::default(),
            submits: Default::default(),
            native_submits: Default::default(),
            waits: Default::default(),
            dispatches: Default::default(),
            compute_passes: Default::default(),

            buffer_allocs: Default::default(),
            bind_group_creates: Default::default(),
            length_buffer_creates: Default::default(),
            readbacks: Default::default(),
            readback_staging_allocs: Cell::new(0),
            memory: poot_runtime_common::MemoryCounters::new(),
            exec_counters: RefCell::new(poot_runtime_common::ExecutionCounters::default()),
            pending_faults: RefCell::new(Vec::new()),
        })
    }

    /// G4: get (or build + cache) the compute pipeline + bind-group layout for `key`, from a
    /// compiler-produced `kernel` (card 608). The number of data buffers is `kernel.args().len()`
    /// (inputs + the one output); the length buffer is bound at slot `n`, and (card 531c) the reserved
    /// error-word buffer at slot `n + 1` when `kernel.has_trap()`. `key` buckets the cache and labels
    /// the wgpu objects; a hit must be built from `kernel`'s exact SPIR-V, binding count and error
    /// word (card 656), so a key naming several kernels holds one pipeline per kernel. Built at most
    /// once per kernel per key per context.
    pub(crate) fn cached_pipeline(
        &self,
        key: &str,
        kernel: &CompiledKernel,
    ) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout), RuntimeError> {
        let poot_runtime_common::KernelCode::SpirvWords(words) = kernel.code() else {
            return Err(RuntimeError::WrongKernelTarget {
                actual: kernel.target(),
            });
        };
        let n = kernel.args().len();
        let has_error_word = kernel.has_trap();
        // Card 656: `key` only buckets the cache; the hit is the pipeline built from this exact module
        // and layout. One plan key can name several kernels (the planner widens a shape-free key's
        // workgroup at a large extent), and a key-only hit dispatched the narrow module.
        if let Some(cp) = self.pipelines.borrow().get(key).and_then(|variants| {
            variants
                .iter()
                .find(|cp| cp.built_from(words, n, has_error_word))
        }) {
            return Ok((cp.pipeline.clone(), cp.bgl.clone()));
        }
        let device = &self.device;
        // Card 158 diagnostic: label the shader module/layout/pipeline with the plan `key` (op+dtype+shape,
        // e.g. "matmul_dq:q4k:[1,1536]:k4096") instead of the generic "kernel"/"bgl"/"pl". wgpu-core error
        // types name resources via `ResourceErrorIdent` ("{type} with '{label}' label"), which feeds the
        // panic text for a deferred `Device::poll` validation error; an unlabeled resource prints an empty
        // label, which is why the card-158 batched-prefill panic (`Validation Error`, wgpu_core.rs:1911)
        // named no op/buffer.
        // SAFETY: wgpu's passthrough contract needs the SPIR-V to be a valid module whose buffer accesses
        // stay in bounds; wgpu checks neither. `kernel` is a `CompiledKernel`, constructible only by
        // poot-codegen's `unsafe` constructor from its own `spirv-val`-clean output, which bounds-checks
        // every slice against the length buffer this crate builds (see the crate doc's "Kernel contract").
        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(key),
                spirv: Some(Cow::Borrowed(words.as_ref())),
                ..Default::default()
            })
        };
        // Match the proven per-dispatch layout: the n data buffers are storage (read_only=false, compatible
        // with the emitted SPIR-V's binding declarations), the length buffer at slot n is read-only. (The
        // kernel only writes the output; marking inputs writable is permitted and matches `dispatch_dev`.)
        // The reserved error-word buffer at slot n+1, when the body has a Trap (card 531c), is writable.
        let mut bgl_entries: Vec<wgpu::BindGroupLayoutEntry> =
            (0..n as u32).map(|i| storage_entry(i, false)).collect();
        bgl_entries.push(storage_entry(n as u32, true));
        if has_error_word {
            bgl_entries.push(storage_entry(n as u32 + 1, false));
        }
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("bgl:{key}")),
            entries: &bgl_entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("pl:{key}")),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.pipeline_builds.set(self.pipeline_builds.get() + 1);
        self.pipelines
            .borrow_mut()
            .entry(key.to_string())
            .or_default()
            .push(CachedPipeline {
                words: words.clone(),
                bindings: n,
                has_error_word,
                pipeline: pipeline.clone(),
                bgl: bgl.clone(),
            });
        Ok((pipeline, bgl))
    }
}

/// The adapter's compute units through [`crate::amd_vulkan_compute_units`]; `None` off Vulkan.
fn probe_amd_compute_units(adapter: &wgpu::Adapter) -> Option<u32> {
    // SAFETY: `as_hal` yields the live wgpu-hal Vulkan adapter for the guard's lifetime, so its
    // instance and physical device are valid for the call.
    unsafe {
        let hal = adapter.as_hal::<wgpu_hal::vulkan::Api>()?;
        crate::amd_vulkan_compute_units(
            hal.shared_instance().raw_instance(),
            hal.raw_physical_device(),
        )
    }
}

/// The launch-shape limits of a wgpu device, converted from what the device was granted and what its
/// adapter reports (the seam [`Context::device_caps`] reads, so a test can feed sentinel values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaunchCaps {
    pub(crate) max_workgroup_size: [u32; 3],
    pub(crate) max_workgroup_invocations: u32,
    pub(crate) subgroup: poot_target::Queried<poot_target::SubgroupSupport>,
}

impl LaunchCaps {
    /// `limits` is the device's granted limits; `subgroup_feature` is whether the adapter supports
    /// `Features::SUBGROUP`; `subgroup_sizes` is the adapter's `(min, max)` subgroup size, which wgpu
    /// fills whether or not the feature exists and so is read only when the feature does.
    pub(crate) fn from_wgpu(
        limits: &wgpu::Limits,
        subgroup_feature: bool,
        subgroup_sizes: (u32, u32),
    ) -> Self {
        Self {
            max_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            max_workgroup_invocations: limits.max_compute_invocations_per_workgroup,
            subgroup: poot_target::Queried::Known(if subgroup_feature {
                poot_target::SubgroupSupport::Present {
                    min_size: subgroup_sizes.0,
                    max_size: subgroup_sizes.1,
                }
            } else {
                poot_target::SubgroupSupport::Absent
            }),
        }
    }
}

#[cfg(test)]
mod launch_caps_tests {
    use super::*;

    /// SC-003: sentinel limits and a subgroup range no real device reports come out exactly, in the
    /// right fields. Mutation: report `downlevel_defaults()` values or swap y and z, and the first
    /// assertion fails with the wrong extents.
    #[test]
    fn launch_caps_carry_the_granted_limits_and_the_adapter_subgroup_range() {
        let limits = wgpu::Limits {
            max_compute_workgroup_size_x: 777,
            max_compute_workgroup_size_y: 333,
            max_compute_workgroup_size_z: 11,
            max_compute_invocations_per_workgroup: 555,
            ..wgpu::Limits::downlevel_defaults()
        };
        let caps = LaunchCaps::from_wgpu(&limits, true, (8, 128));
        assert_eq!(caps.max_workgroup_size, [777, 333, 11]);
        assert_eq!(caps.max_workgroup_invocations, 555);
        assert_eq!(
            caps.subgroup,
            poot_target::Queried::Known(poot_target::SubgroupSupport::Present {
                min_size: 8,
                max_size: 128
            })
        );
    }

    /// A device without the subgroup feature is a confirmed absence, and its adapter size range (which
    /// wgpu fills regardless) is ignored.
    #[test]
    fn an_adapter_without_the_subgroup_feature_reports_absent() {
        let caps = LaunchCaps::from_wgpu(&wgpu::Limits::downlevel_defaults(), false, (8, 128));
        assert_eq!(
            caps.subgroup,
            poot_target::Queried::Known(poot_target::SubgroupSupport::Absent)
        );
    }
}

#[cfg(test)]
mod launch_caps_device_receipt {
    use super::*;

    /// Receipt (SC-003): on a live AMD Vulkan device the launch limits and subgroup range the context
    /// reports equal the ones [`poot_target::DeviceCaps::wgpu_rdna3_igpu`] documents as measured on this
    /// device family. A skip (no GPU, or a non-AMD adapter) is reported as one, not as a pass.
    #[test]
    fn a_live_amd_device_reports_the_launch_limits_the_fixture_documents() {
        let Ok(ctx) = Context::new() else {
            eprintln!(
                "SKIP a_live_amd_device_reports_the_launch_limits_the_fixture_documents: no GPU"
            );
            return;
        };
        if !ctx.is_amd_radv {
            eprintln!(
                "SKIP a_live_amd_device_reports_the_launch_limits_the_fixture_documents: not RADV"
            );
            return;
        }
        let live = ctx.device_caps();
        eprintln!("live wgpu caps: {live:?}");
        let fixture = poot_target::DeviceCaps::wgpu_rdna3_igpu();
        assert_eq!(live.max_workgroup_size, fixture.max_workgroup_size);
        assert_eq!(
            live.max_workgroup_invocations,
            fixture.max_workgroup_invocations
        );
        assert_eq!(live.subgroup, fixture.subgroup);
    }
}

#[cfg(test)]
mod card537_wgpu_debug_tests {
    use super::*;

    /// SC-003: with `POOT_WGPU_DEBUG` set, an injected wgpu validation error is still fatal -
    /// the custom `on_uncaptured_error` handler decorates it with diagnostics and re-panics, it never
    /// swallows it into silent continuation. Skips cleanly without a wgpu device.
    ///
    /// Mutation (recorded here, never left in the tree): removing the trailing `panic!` from the
    /// `on_uncaptured_error` closure above (the R482-013 bug: log the error, then continue) made this
    /// test observe no panic across the injected out-of-range write - red (`caught.is_err()` was
    /// `false`, the write silently continued); restoring the `panic!` made it green.
    #[test]
    fn wgpu_debug_never_downgrades_a_fatal_uncaptured_error() {
        // SAFETY: test-only env mutation of a process-global; cargo-nextest gives every #[test] its own
        // process (poot-gpu's card 537 tests use the same pattern). Must be set before `Context::new()`:
        // the read happens once, at construction.
        unsafe {
            std::env::set_var("POOT_WGPU_DEBUG", "1");
        }
        let ctx = match Context::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!(
                    "SKIP wgpu_debug_never_downgrades_a_fatal_uncaptured_error: no GPU ({e})"
                );
                // SAFETY: see the set_var above.
                unsafe {
                    std::env::remove_var("POOT_WGPU_DEBUG");
                }
                return;
            }
        };
        // A 4-byte buffer with a 16-byte write is out of range: wgpu validates `write_buffer`'s size
        // against the destination buffer and routes the violation through `on_uncaptured_error`, not a
        // `Result` (unlike this crate's own `write_f32`/`write_bytes`, which range-check first).
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("card537-sc003-undersized"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ctx.queue.write_buffer(&buf, 0, &[0u8; 16]);
            let _ = ctx.poll_wait();
        }));
        // SAFETY: see the set_var above.
        unsafe {
            std::env::remove_var("POOT_WGPU_DEBUG");
        }
        assert!(
            caught.is_err(),
            "an out-of-range write_buffer must still panic with POOT_WGPU_DEBUG set"
        );
    }
}
