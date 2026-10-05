use crate::*;

/// [`Context::build_dispatch_objects`]'s result: the pipeline, bind group, resolved grid, the
/// optional error-word buffer, and the role-tagged live-bytes guards for the length buffer and (when
/// present) the error-word buffer (Card 547a SC-005).
type DispatchObjects = (
    wgpu::ComputePipeline,
    wgpu::BindGroup,
    [u32; 3],
    Option<wgpu::Buffer>,
    std::sync::Arc<poot_runtime_common::AllocGuard>,
    Option<std::sync::Arc<poot_runtime_common::AllocGuard>>,
);

impl Context {
    /// Stage an asserting dispatch's error-word storage buffer as a [`PendingFault`] instead of reading
    /// it back now (card 531c): the no-per-dispatch-sync paths (`dispatch_dev`,
    /// `submit_cached`) call this right after building the buffer; it is read back
    /// and checked at the next call that already syncs with the device (`Self::stage_pending_faults` /
    /// `Self::check_staged_faults`). `guard` is `Some` only when `storage` was freshly allocated right
    /// here and needs its own charge (`dispatch_dev`); `submit_cached` restages an existing
    /// `CachedDispatch.error_storage`, already charged for its whole lifetime, so it passes `None`
    /// (Card 547a).
    pub(crate) fn record_pending_fault(
        &self,
        label: &str,
        storage: wgpu::Buffer,
        guard: Option<std::sync::Arc<poot_runtime_common::AllocGuard>>,
    ) {
        self.pending_faults.borrow_mut().push(PendingFault {
            label: label.to_string(),
            storage,
            guard,
        });
    }

    /// Drain every staged [`PendingFault`] and copy each one's error word into its own small mappable
    /// staging buffer, as copy commands added to `enc` - the caller's own encoder, about to be submitted
    /// in the same `queue.submit` the caller was already making. Returns the staged buffers (each with
    /// its own Staging-role live-bytes guard, Card 547a: this is direct validation staging by
    /// name, so `MemoryCounters` must see it) so the caller can `map_async` them alongside its own
    /// buffer(s) before polling (card 531c: this adds no new submit or poll, it rides
    /// the caller's existing one).
    pub(crate) fn stage_pending_faults(
        &self,
        enc: &mut wgpu::CommandEncoder,
    ) -> Vec<(
        String,
        wgpu::Buffer,
        std::sync::Arc<poot_runtime_common::AllocGuard>,
    )> {
        self.pending_faults
            .borrow_mut()
            .drain(..)
            .map(|f| {
                let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("error-word-staging"),
                    size: 4,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let guard = self.tag_alloc(BufferRole::Staging, 4);
                enc.copy_buffer_to_buffer(&f.storage, 0, &staging, 0, 4);
                (f.label, staging, guard)
            })
            .collect()
    }

    /// Read every buffer [`Self::stage_pending_faults`] returned - already mapped by the caller's own
    /// `map_async` + `poll(Wait)`, which waits for the whole submission, not only the buffers it names
    /// explicitly - and raise the first nonzero one as [`RuntimeError::KernelAssertFailed`]. Card 531c: must run before the caller returns `Ok`.
    pub(crate) fn check_staged_faults(
        &self,
        staged: &[(
            String,
            wgpu::Buffer,
            std::sync::Arc<poot_runtime_common::AllocGuard>,
        )],
    ) -> Result<(), RuntimeError> {
        for (label, staging, _) in staged {
            let code = {
                let data = staging.slice(..).get_mapped_range();
                bytemuck::cast_slice::<u8, u32>(&data)[0]
            };
            if code != 0 {
                return Err(RuntimeError::KernelAssertFailed {
                    kernel: label.clone(),
                    code,
                });
            }
        }
        Ok(())
    }

    /// Build (once) the pipeline + length buffer + bind group + workgroup-count grid for one dispatch
    /// over caller-supplied buffer handles and a compiler-produced `kernel` (card 608). Used by
    /// [`Context::build_cached_dispatch`] (card 156 phase 3/4: once at cache-build time, never per
    /// step). Counts every `create_buffer_init` (the length buffer) and `create_bind_group` call so
    /// callers can assert a warm cached path makes none per step (SC-001); `cached_pipeline` counts
    /// pipeline builds separately.
    ///
    /// Checks `ins`/`out` against `kernel`'s argument schema (card 608, SC-002) before building anything.
    /// When `kernel.has_trap()` this binds and zeroes the reserved error-word buffer, since the compiled
    /// module declares that binding, and returns it as the 4th tuple element. This function does not
    /// read it back here; [`Context::build_cached_dispatch`] instead stores it in the returned
    /// [`CachedDispatch`], since that buffer is reused across every step and must be re-zeroed and
    /// re-registered by [`Context::submit_cached`] each time, not once at build time (card 531c).
    ///
    /// `ins_lens`/`out_len` (Card 547b) are each buffer's own logical device-element
    /// count from the plan, index-aligned with `ins` plus `out` last: the length buffer this builds
    /// carries these, never `DeviceBuffer::elem_count()` (an arena slot's buffer is sized to its
    /// largest occupant, which can exceed this particular dispatch's operand).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_dispatch_objects(
        &self,
        key: &str,
        kernel: &CompiledKernel,
        wg: [u32; 3],
        threads: [u32; 3],
        ins: &[&DeviceBuffer],
        ins_lens: &[u32],
        out: &DeviceBuffer,
        out_len: u32,
    ) -> Result<DispatchObjects, RuntimeError> {
        check_device_args(kernel, ins, out)?;
        debug_assert_eq!(
            ins.len(),
            ins_lens.len(),
            "ins_lens must be index-aligned with ins"
        );
        let n = ins.len() + 1;
        let needs_error_word = kernel.has_trap();
        // Fold first so the length buffer can carry the X-thread extent the kernel uses to rebuild
        // the linear index under a 2-D launch (shared codegen, see poot-codegen emit_thread_index).
        let groups = resolve_groups(threads, wg, self.max_workgroups)?;
        let (pipeline, bgl) = self.cached_pipeline(key, kernel)?;
        let mut lengths: Vec<u32> = ins_lens.to_vec();
        lengths.push(out_len);
        lengths.push(x_extent(groups, wg));
        let len_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                // Card 158 diagnostic: label from the key (see the comment on `cached_pipeline`), not the generic
                // "lengths".
                label: Some(&format!("lengths:{key}")),
                contents: bytemuck::cast_slice(&lengths),
                usage: wgpu::BufferUsages::STORAGE,
            });
        self.length_buffer_creates
            .set(self.length_buffer_creates.get() + 1);
        // Card 547a (SC-005): the length buffer is direct validation-adjacent staging the one memory
        // service must still see, not only `alloc_f32`/`alloc_storage`'s own callers. Its guard is
        // threaded into the returned `CachedDispatch`, so it stays live exactly as long as the
        // `bind_group` that references it (not dropped at the end of this function).
        let length_buffer_guard = self.tag_alloc(
            poot_runtime_common::BufferRole::Staging,
            std::mem::size_of_val(lengths.as_slice()),
        );
        // Error word: one u32, zeroed (card 531c); the caller decides how/when to check it back, see
        // this function's doc comment. COPY_DST: `Context::submit_cached` re-zeroes this same buffer
        // with `queue.write_buffer` on every later step when this dispatch is cached (card 531c).
        let err_storage = needs_error_word.then(|| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("error-word:{key}")),
                    contents: bytemuck::cast_slice(&[0u32]),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                })
        });
        // Card 547a (SC-005): same accounting for the error-word buffer, when this kernel has one.
        let error_storage_guard =
            needs_error_word.then(|| self.tag_alloc(poot_runtime_common::BufferRole::Staging, 4));
        let mut entries: Vec<wgpu::BindGroupEntry> = ins
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.buf.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: ins.len() as u32,
            resource: out.buf.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: n as u32,
            resource: len_buf.as_entire_binding(),
        });
        if let Some(err) = &err_storage {
            entries.push(wgpu::BindGroupEntry {
                binding: n as u32 + 1,
                resource: err.as_entire_binding(),
            });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            // Card 158 diagnostic: label from the key, not the generic "bg".
            label: Some(&format!("bg:{key}")),
            layout: &bgl,
            entries: &entries,
        });
        self.bind_group_creates
            .set(self.bind_group_creates.get() + 1);
        Ok((
            pipeline,
            bind_group,
            groups,
            err_storage,
            length_buffer_guard,
            error_storage_guard,
        ))
    }

    /// Card 156 phase 3/4: build (once) the pipeline + bind group + grid for one dispatch over stable
    /// buffer handles: the pooled output buffer, the ping-pong KV pair member for this parity, and the
    /// persistent Slot buffers. The length buffer baked into the bind group is also built only here (every
    /// `elem_count` is step-invariant for a fixed-shape decode graph), rebuilt fresh per dispatch per call
    /// elsewhere (SF-2). The caller (poot-gpu's `DecodeCache`) holds the returned [`CachedDispatch`] and
    /// re-encodes it every step via
    /// [`Context::submit_cached`].
    #[allow(clippy::too_many_arguments)]
    pub fn build_cached_dispatch(
        &self,
        label: &str,
        key: &str,
        kernel: &CompiledKernel,
        wg: [u32; 3],
        threads: [u32; 3],
        ins: &[&DeviceBuffer],
        ins_lens: &[u32],
        out: &DeviceBuffer,
        out_len: u32,
    ) -> Result<CachedDispatch, RuntimeError> {
        let (pipeline, bind_group, groups, error_storage, length_buffer_guard, error_storage_guard) =
            self.build_dispatch_objects(key, kernel, wg, threads, ins, ins_lens, out, out_len)?;
        Ok(CachedDispatch {
            pipeline,
            bind_group,
            groups,
            label: label.to_string(),
            error_storage,
            length_buffer_guard,
            error_storage_guard,
        })
    }

    /// Card 156 phase 4: re-encode + submit one token's worth of already-resolved [`CachedDispatch`]es.
    /// No pipeline, length-buffer, or bind-group work happens here: a fresh `CommandEncoder`, one compute
    /// pass for a nonempty unattributed batch (one per dispatch under the Card 552 typed device-timing
    /// path), and one `queue.submit`. Existing batch boundaries and caller-owned drains are unchanged.
    ///
    /// A `CachedDispatch` with an error word (`error_storage`) reuses that same buffer every step, so it
    /// must be re-zeroed here before this step's dispatch (`queue.write_buffer`, no extra sync - the
    /// write is just enqueued) and re-registered as a
    /// [`PendingFault`] on every call, not once at build time (card 531c).
    ///
    /// Card 600 (SC-003): hands `submit_encoded` a lazy [`EncodeItem`] iterator directly instead of
    /// collecting one, so a replayed step makes no host allocation here - `dispatches` is already a
    /// caller-held `&[&CachedDispatch]`, and `EncodeItem` is only ever a borrowed view over it. This
    /// does not reach every dispatch, though: a trap-bearing one still costs one `label.to_string()`
    /// per replayed step, inside `record_pending_fault` above (review F5) - a known remaining
    /// per-dispatch allocation, out of SC-003's measured surface (its fixture has no trap kernel).
    pub fn submit_cached(&self, dispatches: &[&CachedDispatch]) -> Result<(), RuntimeError> {
        for d in dispatches {
            if let Some(err) = &d.error_storage {
                self.queue
                    .write_buffer(err, 0, bytemuck::cast_slice(&[0u32]));
                self.record_pending_fault(&d.label, err.clone(), None);
            }
        }
        self.submit_encoded(dispatches.iter().map(|d| EncodeItem {
            pipeline: &d.pipeline,
            bind_group: &d.bind_group,
            groups: d.groups,
            label: &d.label,
        }))
    }

    /// Card 552 review F2/F3: fold whatever is still pending into the running accumulator, return
    /// its total, and reset both for the next step. Never polls itself (the caller - `Engine::step`,
    /// after its own `synchronize()` - must already have made every pending `map_async` resolve).
    /// `None` only when nothing was ever collected this step (counters-only mode, no `Device`
    /// capability, or a step whose replay dispatched nothing) - never a fabricated zero.
    pub fn drain_device_timing(&self) -> Option<DeviceTimingResult> {
        let pending: Vec<PendingDeviceTiming> =
            self.pending_device_timing.borrow_mut().drain(..).collect();
        let period = self.queue.get_timestamp_period() as f64;
        let mut acc = self.drained_device_timing.take();
        for p in &pending {
            acc.fold(p, period);
        }
        self.next_dispatch_index.set(0);
        if acc.per_dispatch.is_empty() {
            return None;
        }
        let span = match (acc.min_start, acc.max_end) {
            (Some(s), Some(e)) => {
                Duration::from_nanos((e.saturating_sub(s) as f64 * period) as u64)
            }
            _ => Duration::ZERO,
        };
        Some(DeviceTimingResult {
            sum: acc.sum,
            span,
            per_dispatch: acc.per_dispatch,
        })
    }

    /// Card 552: enforce the declared in-flight-query bound before registering one more
    /// pending readback. `0` means unbounded. Past the bound, wait for and fold the oldest pending
    /// entries first (`DrainPolicy::WaitForCapacity`, the only policy today) - an explicit,
    /// declared synchronization this configuration asked for, never a silent one.
    fn drain_for_capacity(&self) -> Result<(), RuntimeError> {
        let bound = self.max_in_flight_queries.get();
        if bound == 0 {
            return Ok(());
        }
        while self.pending_device_timing.borrow().len() >= bound {
            if self.pending_device_timing.borrow().is_empty() {
                break;
            }
            let oldest = self.pending_device_timing.borrow_mut().remove(0);
            self.waits.set(self.waits.get() + 1);
            self.record_wait(poot_runtime_common::CallPurpose::Timing);
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .map_err(|e| RuntimeError::Readback(e.to_string()))?;
            let period = self.queue.get_timestamp_period() as f64;
            self.drained_device_timing
                .borrow_mut()
                .fold(&oldest, period);
        }
        Ok(())
    }

    /// Encode and submit one ordered batch of already-resolved dispatches. Unattributed nonempty
    /// batches share one compute pass; the Card 552 typed device-timing path keeps one pass per
    /// dispatch for its timestamp contract (review F1: one submit path, not a second encoder). wgpu
    /// tracks compute resource usage and inserts dependency barriers per dispatch within a pass.
    /// Without timestamp attribution, an empty batch still submits an empty encoder with no compute
    /// pass.
    ///
    /// Card 600 (SC-003): `items` is a lazy, caller-built [`EncodeItem`] iterator (never a `Vec`), so
    /// this function and [`Context::submit_cached`] make no host allocation of their own per replayed
    /// step.
    pub(crate) fn submit_encoded<'a>(
        &self,
        items: impl ExactSizeIterator<Item = EncodeItem<'a>> + Clone,
    ) -> Result<(), RuntimeError> {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("poot-batch-encoder"),
            });

        // Attributed path (the Card 552 typed device-timing collector): a timestamp query set with 2
        // slots per dispatch, one pass each, so device time is attributed per dispatch. Off ->
        // `query_set` stays `None` and the single-pass branch below runs exactly as before Card 552.
        let attributed = self.device_timing.get() && self.timestamps;
        let dispatch_count = items.len();
        let query_set = (attributed && self.timestamps).then(|| {
            self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("poot-ts-batch"),
                ty: wgpu::QueryType::Timestamp,
                count: (dispatch_count * 2) as u32,
            })
        });

        // Preserve every pipeline, binding, grid and dispatch in order. wgpu 29's compute usage
        // scope is per dispatch, so dependent items do not require separate pass boundaries.
        // An attributed collector without timestamp support still takes the per-dispatch path.
        if !attributed && dispatch_count != 0 {
            self.compute_passes.set(self.compute_passes.get() + 1);
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("poot-batch"),
                timestamp_writes: None,
            });
            for e in items {
                cpass.set_pipeline(e.pipeline);
                cpass.set_bind_group(0, e.bind_group, &[]);
                self.dispatches.set(self.dispatches.get() + 1);
                cpass.dispatch_workgroups(e.groups[0], e.groups[1], e.groups[2]);
            }
        } else {
            for (i, e) in items.enumerate() {
                let timestamp_writes =
                    query_set
                        .as_ref()
                        .map(|qs| wgpu::ComputePassTimestampWrites {
                            query_set: qs,
                            beginning_of_pass_write_index: Some((i * 2) as u32),
                            end_of_pass_write_index: Some((i * 2 + 1) as u32),
                        });
                // Card 158 diagnostic: label the pass with the dispatch's label (the plan key on the resident
                // path); some backends show compute-pass boundaries in validation traces (e.g. Vulkan debug-utils
                // command-buffer labels), so the failing op's name can surface here too.
                self.compute_passes.set(self.compute_passes.get() + 1);
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(e.label),
                    timestamp_writes,
                });
                cpass.set_pipeline(e.pipeline);
                cpass.set_bind_group(0, e.bind_group, &[]);
                self.dispatches.set(self.dispatches.get() + 1);
                cpass.dispatch_workgroups(e.groups[0], e.groups[1], e.groups[2]);
            }
        }

        // Resolve the timestamps (attributed only) into a mappable buffer before submit.
        let (ts_read, ts_guard) = if let Some(qs) = &query_set {
            let count = (dispatch_count * 2) as u64;
            let bytes = count * 8;
            let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("poot-ts-resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("poot-ts-read"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            // Card 547a: charge both batched timestamp buffers together as Staging.
            let guard = self.tag_alloc(BufferRole::Staging, (bytes * 2) as usize);
            enc.resolve_query_set(qs, 0..(dispatch_count * 2) as u32, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &read, 0, bytes);
            (Some(read), Some(guard))
        } else {
            (None, None)
        };

        self.native_submits.set(self.native_submits.get() + 1);
        self.record_submit(poot_runtime_common::CallPurpose::Compute);
        self.queue.submit(Some(enc.finish()));
        self.submits.set(self.submits.get() + 1);

        if let Some(read) = ts_read {
            // `ts_read` is `Some` only when `query_set` was built, which required `attributed` (i.e.
            // `self.device_timing.get()`) above - register the map without polling. The caller's
            // later `synchronize()` (or an earlier chunk-boundary `poll_wait`, or this function's own
            // F3 capacity drain on a later call) satisfies it too, so collecting this detail adds no
            // synchronization beyond what the step already declares/does.
            self.drain_for_capacity()?;
            let base_index = self.next_dispatch_index.get();
            self.next_dispatch_index.set(base_index + dispatch_count);
            read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            self.pending_device_timing
                .borrow_mut()
                .push(PendingDeviceTiming {
                    read,
                    base_index,
                    count: dispatch_count,
                    guard: ts_guard.expect("query_set implies a Staging guard"),
                });
        }
        Ok(())
    }

    /// Dispatch a compiler-produced `kernel` (card 608). `wg` is the workgroup shape (must match the
    /// module's `LocalSize`), `threads` the per-axis thread counts. `buffers` are checked against `kernel`'s
    /// argument schema before anything is built (SC-002); writable buffers are read back into their
    /// bytes. When `kernel.has_trap()` the compiled module declares one more binding after the length
    /// buffer, which this call zeroes before dispatch and checks after the existing sync, turning a
    /// nonzero code into [`RuntimeError::KernelAssertFailed`].
    pub fn dispatch(
        &self,
        label: &str,
        kernel: &CompiledKernel,
        wg: [u32; 3],
        threads: [u32; 3],
        buffers: &mut [KernelBuffer],
    ) -> Result<(), RuntimeError> {
        check_kernel_buffer_args(kernel, buffers)?;
        let words = spirv_words(kernel)?;
        let needs_error_word = kernel.has_trap();
        let device = &self.device;
        let n = buffers.len();

        // SAFETY: the same caller obligation as `cached_pipeline`'s passthrough (the crate doc's
        // "Kernel contract"): `kernel` is a `CompiledKernel`, constructible only by poot-codegen's
        // `unsafe` constructor from its own bounds-guarded SPIR-V output. Not type-enforced beyond that.
        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("kernel"),
                spirv: Some(Cow::Borrowed(words)),
                ..Default::default()
            })
        };
        // All data slots are writable in the descriptor (the emitter emits writable buffers); only the
        // length buffer is read-only. The error word (if any), right after it, is writable (the kernel
        // atomically writes its trap code there).
        let mut bgl_entries: Vec<wgpu::BindGroupLayoutEntry> =
            (0..n as u32).map(|i| storage_entry(i, false)).collect();
        bgl_entries.push(storage_entry(n as u32, true));
        if needs_error_word {
            bgl_entries.push(storage_entry(n as u32 + 1, false));
        }
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &bgl_entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        self.pipeline_builds.set(self.pipeline_builds.get() + 1);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("kernel"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Storage buffers + staging buffers for writable slots.
        let mut storage_bufs = Vec::with_capacity(n);
        let mut staging: Vec<Option<wgpu::Buffer>> = Vec::with_capacity(n);
        // Card 547a: each writable slot's own MAP_READ result staging buffer, charged
        // Staging, held until the readback below completes (end of this function, same scope as
        // `staging` itself).
        let mut _staging_guards: Vec<Option<std::sync::Arc<poot_runtime_common::AllocGuard>>> =
            Vec::with_capacity(n);
        for (i, b) in buffers.iter().enumerate() {
            let usage = if b.writable() {
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
            } else {
                wgpu::BufferUsages::STORAGE
            };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("slot{i}")),
                contents: b.bytes(),
                usage,
            });
            if b.writable() {
                let size = b.bytes().len() as wgpu::BufferAddress;
                let s = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("staging{i}")),
                    size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                _staging_guards.push(Some(self.tag_alloc(BufferRole::Staging, size as usize)));
                staging.push(Some(s));
            } else {
                _staging_guards.push(None);
                staging.push(None);
            }
            storage_bufs.push(buf);
        }
        // length buffer [u32; n+1] at binding n: one slot per data buffer, then the folded X-thread
        // extent for `thread_index` reconstruction (shared codegen).
        let groups = resolve_groups(threads, wg, self.max_workgroups)?;
        let mut lengths: Vec<u32> = buffers.iter().map(KernelBuffer::elem_count).collect();
        lengths.push(x_extent(groups, wg));
        self.length_buffer_creates
            .set(self.length_buffer_creates.get() + 1);
        let len_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lengths"),
            contents: bytemuck::cast_slice(&lengths),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // Card 547a: same direct-allocation accounting `build_dispatch_objects` already
        // gives its own length buffer.
        let _len_buf_guard = self.tag_alloc(
            BufferRole::Staging,
            std::mem::size_of_val(lengths.as_slice()),
        );
        // Error word: one u32, zeroed before every dispatch (0 means "no fault"); the kernel's Trap
        // lowering atomically writes its trap code there instead of a device abort (card 531c).
        let err_storage = needs_error_word.then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("error-word"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        });
        // Card 547a: same accounting as `build_dispatch_objects`'s `error_storage_guard`.
        let _err_storage_guard = needs_error_word.then(|| self.tag_alloc(BufferRole::Staging, 4));
        let err_staging = needs_error_word.then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("error-word-staging"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        // Card 547a: this call's own error-word readback staging, like
        // `stage_pending_faults`'s per-fault buffer.
        let _err_staging_guard = needs_error_word.then(|| self.tag_alloc(BufferRole::Staging, 4));

        let mut bg_entries: Vec<wgpu::BindGroupEntry> = storage_bufs
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        bg_entries.push(wgpu::BindGroupEntry {
            binding: n as u32,
            resource: len_buf.as_entire_binding(),
        });
        if let Some(err) = &err_storage {
            bg_entries.push(wgpu::BindGroupEntry {
                binding: n as u32 + 1,
                resource: err.as_entire_binding(),
            });
        }
        self.bind_group_creates
            .set(self.bind_group_creates.get() + 1);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bgl,
            entries: &bg_entries,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            self.compute_passes.set(self.compute_passes.get() + 1);
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            self.dispatches.set(self.dispatches.get() + 1);
            cpass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        for (i, s) in staging.iter().enumerate() {
            if let Some(stg) = s {
                encoder.copy_buffer_to_buffer(&storage_bufs[i], 0, stg, 0, storage_bufs[i].size());
            }
        }
        if let (Some(err), Some(stg)) = (&err_storage, &err_staging) {
            encoder.copy_buffer_to_buffer(err, 0, stg, 0, 4);
        }
        // Card 531c: this call already does a full submit+map+poll cycle, so it also
        // flushes any fault a preceding `dispatch_dev`/`submit_cached` call staged
        // and hasn't been checked yet (a caller may mix this cold path into an otherwise resident step).
        let staged_faults = self.stage_pending_faults(&mut encoder);
        self.native_submits.set(self.native_submits.get() + 1);
        // Card 552 SC-007: this one submission also carries the writable-buffer/error-word/
        // timestamp readback copies queued above - a known mixed-purpose call, counted once under
        // its one dominant purpose (the compute dispatch), never split or duplicated across
        // purposes.
        self.record_submit(poot_runtime_common::CallPurpose::Compute);
        self.queue.submit(Some(encoder.finish()));
        self.submits.set(self.submits.get() + 1); // one compute submit (per-op dispatch path)

        for s in staging.iter().flatten() {
            s.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        if let Some(stg) = &err_staging {
            stg.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        for (_, fs, _) in &staged_faults {
            fs.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        self.waits.set(self.waits.get() + 1);
        // Card 552 SC-007: one dominant purpose (reading results back), same reasoning as the
        // submission above.
        self.record_wait(poot_runtime_common::CallPurpose::Readback);
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RuntimeError::Readback(e.to_string()))?;
        for (stg, b) in staging.iter().zip(buffers.iter_mut()) {
            if let Some(stg) = stg {
                let data = stg.slice(..).get_mapped_range();
                b.overwrite_after_readback(&data);
            }
        }
        self.check_staged_faults(&staged_faults)?;
        if let Some(stg) = &err_staging {
            let code = {
                let data = stg.slice(..).get_mapped_range();
                bytemuck::cast_slice::<u8, u32>(&data)[0]
            };
            if code != 0 {
                return Err(RuntimeError::KernelAssertFailed {
                    kernel: label.to_string(),
                    code,
                });
            }
        }
        // Host round-trip path: every buffer is uploaded, writable buffers are read back.
        let h2d: usize = buffers.iter().map(|b| b.bytes().len()).sum();
        let d2h: usize = buffers
            .iter()
            .filter(|b| b.writable())
            .map(|b| b.bytes().len())
            .sum();
        self.record_transfer(
            poot_runtime_common::CallPurpose::Upload,
            poot_runtime_common::TransferDirection::HostToDevice,
            h2d as u64,
        );
        self.record_transfer(
            poot_runtime_common::CallPurpose::Readback,
            poot_runtime_common::TransferDirection::DeviceToHost,
            d2h as u64,
        );
        Ok(())
    }

    /// Typed one-shot launch: dispatch `kernel` with read-only f32 `inputs` bound first (in order) and a
    /// single writable f32 `output` bound last, one thread per output element (64-wide workgroups); the
    /// result is copied back into `output`. The entry the `#[kernel]` host wrapper calls; the kernel's
    /// `&mut` output slice must be its last parameter (the binding convention).
    ///
    /// This calls [`Context::dispatch`] beneath it, so a `#[kernel]` source whose body has an
    /// `Assert`/`Unreachable` (`kernel.has_trap()` is true) is handled the same way: the host wrapper's
    /// handle comes from `poot_codegen::kernel_handle` against the device pass's own compiled `Body` (card
    /// 608), so `has_trap` here is never guessed from the Rust signature.
    pub fn launch(
        &self,
        kernel: &CompiledKernel,
        inputs: &[&[f32]],
        output: &mut [f32],
    ) -> Result<(), RuntimeError> {
        let mut bufs: Vec<KernelBuffer> = inputs
            .iter()
            .map(|d| KernelBuffer::read_only_f32(d))
            .collect();
        bufs.push(KernelBuffer::write_f32(output.len()));
        self.dispatch(
            "launch",
            kernel,
            [64, 1, 1],
            [output.len() as u32, 1, 1],
            &mut bufs,
        )?;
        output.copy_from_slice(bufs.last().unwrap().as_f32());
        Ok(())
    }

    /// Dispatch a compiler-produced `kernel` (card 608) reading the persistent `ins` buffers and writing
    /// `out` (which must already exist). `ins`/`out` are checked against `kernel`'s argument schema
    /// before anything is built (SC-002). The pipeline is `cached_pipeline`'s (built once per
    /// `label` and kernel); the `[u32;N]` length buffer (one per slot, ins then out) and the bind group are
    /// built per call.
    ///
    /// When `kernel.has_trap()` this binds and zeroes the reserved error-word buffer, since the compiled
    /// module declares that binding. This call does **not** read the word back itself - this is the
    /// device-resident fast path with no per-dispatch host sync, and adding one here would defeat it - it
    /// is instead staged as a pending fault and read back at the next call that already syncs with the
    /// device (card 531c).
    pub fn dispatch_dev(
        &self,
        label: &str,
        kernel: &CompiledKernel,
        wg: [u32; 3],
        threads: [u32; 3],
        ins: &[&DeviceBuffer],
        out: &DeviceBuffer,
    ) -> Result<(), RuntimeError> {
        check_device_args(kernel, ins, out)?;
        let needs_error_word = kernel.has_trap();
        let device = &self.device;
        let n = ins.len() + 1; // data slots
        // Card 657: the pipeline comes from the context's cache (keyed by `label`, hit only on this exact
        // kernel), so a per-step caller builds it once, not on every call.
        let (pipeline, bgl) = self.cached_pipeline(label, kernel)?;
        let groups = resolve_groups(threads, wg, self.max_workgroups)?;
        let mut lengths: Vec<u32> = ins.iter().map(|b| b.elem_count()).collect();
        lengths.push(out.elem_count());
        lengths.push(x_extent(groups, wg));
        self.length_buffer_creates
            .set(self.length_buffer_creates.get() + 1);
        let len_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lengths"),
            contents: bytemuck::cast_slice(&lengths),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // Card 547a: same direct-allocation accounting `build_dispatch_objects` already
        // gives its own length buffer.
        let _len_buf_guard = self.tag_alloc(
            BufferRole::Staging,
            std::mem::size_of_val(lengths.as_slice()),
        );
        let err_storage = needs_error_word.then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("error-word"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        });
        // Card 547a: this one-shot error word is moved into `record_pending_fault` below,
        // not dropped here, so its guard travels with it (unlike `build_dispatch_objects`'s, which
        // travels inside a `CachedDispatch` instead).
        let err_storage_guard = needs_error_word.then(|| self.tag_alloc(BufferRole::Staging, 4));
        let mut entries: Vec<wgpu::BindGroupEntry> = ins
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.buf.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: ins.len() as u32,
            resource: out.buf.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: n as u32,
            resource: len_buf.as_entire_binding(),
        });
        if let Some(err) = &err_storage {
            entries.push(wgpu::BindGroupEntry {
                binding: n as u32 + 1,
                resource: err.as_entire_binding(),
            });
        }
        self.bind_group_creates
            .set(self.bind_group_creates.get() + 1);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bgl,
            entries: &entries,
        });
        // Freshly created and zeroed above, so (unlike `submit_cached`'s reused buffer) no re-zero is
        // needed here: stage it as pending, read back and checked at the next call that already syncs
        // with the device (card 531c).
        if let Some(err) = err_storage {
            self.record_pending_fault(label, err, err_storage_guard);
        }
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            self.compute_passes.set(self.compute_passes.get() + 1);
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            self.dispatches.set(self.dispatches.get() + 1);
            cpass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        self.native_submits.set(self.native_submits.get() + 1);
        self.record_submit(poot_runtime_common::CallPurpose::Compute);
        self.queue.submit(Some(enc.finish()));
        self.submits.set(self.submits.get() + 1); // one compute submit (resident per-dispatch path)
        // Returns without a sync: the device-resident fast path leaves results on the GPU, no
        // per-dispatch host wait.
        Ok(())
    }
}

/// Card 600 acceptance (SC-001/SC-002/SC-003): a replayed step's `submit_cached`/`submit_encoded`
/// path creates no device object and makes no host allocation past the one-time
/// `build_cached_dispatch` that builds each dispatch's pipeline, length buffer and bind group.
#[cfg(test)]
mod step_encode_allocation_tests {
    use crate::{CachedDispatch, CompiledKernel, Context, DeviceBuffer, EncodeItem};
    use poot_kernel_ir::fixtures;
    use poot_runtime_common::DeviceBackend;
    use poot_test_util::device_skip::open_or_skip;

    fn kernel(body: &poot_kernel_ir::Body, name: &str) -> CompiledKernel {
        let dir = std::env::temp_dir().join("poot-runtime-dispatch-alloc-tests");
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

    /// 4 add dispatches sharing one kernel/key but each over its own distinct input/output buffers -
    /// shaped like a dense decode step's batch of independent per-layer dispatches.
    fn four_dispatches(ctx: &Context) -> (Vec<CachedDispatch>, Vec<DeviceBuffer>) {
        let add = kernel(&fixtures::add_kernel(), "sc600-add");
        let a: Vec<_> = (0..4)
            .map(|i| ctx.upload_f32(&[(i + 1) as f32; 17]))
            .collect();
        let b: Vec<_> = (0..4)
            .map(|i| ctx.upload_f32(&[(i + 10) as f32; 17]))
            .collect();
        let outputs: Vec<_> = (0..4).map(|_| ctx.alloc_f32(17)).collect();
        let dispatches: Vec<_> = (0..4)
            .map(|i| {
                cached(
                    ctx,
                    &format!("sc600-{i}"),
                    "sc600-add",
                    &add,
                    &[&a[i], &b[i]],
                    &outputs[i],
                    17,
                )
            })
            .collect();
        (dispatches, outputs)
    }

    /// SC-001 - after the dispatches are built once (`build_cached_dispatch`: the "load + one
    /// warm-up step" half of a step's lifecycle), 16 further `submit_cached` calls ("steps") create
    /// no further bind group, pipeline or length buffer.
    ///
    /// MUTATION (recorded here, not left in the tree; Card 600 SC-001): add
    /// `self.bind_group_creates.set(self.bind_group_creates.get() + dispatches.len());` at the top
    /// of `Context::submit_cached`'s body, simulating a bind group rebuilt per dispatch on every
    /// replay. Result: RED - `bind_group_creates` reads delta 64 (4 dispatches * 16 steps) over the
    /// measured steps instead of 0. Reverted: GREEN (both readings recorded in the card).
    #[test]
    fn dense_decode_replay_creates_no_bind_groups_pipelines_or_length_buffers_gpu() {
        let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
            return;
        };
        let (dispatches, _outputs) = four_dispatches(&ctx);
        let refs: Vec<&CachedDispatch> = dispatches.iter().collect();
        let before = ctx.counters().snapshot();
        for _ in 0..16 {
            ctx.submit_cached(&refs).unwrap();
        }
        let after = ctx.counters().snapshot();
        assert_eq!(
            after.pipeline_builds, before.pipeline_builds,
            "pipeline_builds must stay flat over 16 replayed steps"
        );
        assert_eq!(
            after.bind_group_creates, before.bind_group_creates,
            "bind_group_creates must stay flat over 16 replayed steps"
        );
        assert_eq!(
            after.length_buffer_creates, before.length_buffer_creates,
            "length_buffer_creates must stay flat over 16 replayed steps"
        );
    }

    /// SC-002 - a replay with distinct per-dispatch buffers produces exact results (integer-valued
    /// inputs make the expected sums exact in f32, so this is bit-for-bit `assert_eq!`, never a
    /// tolerance): every dispatch binds its own buffers, not a neighbor's.
    ///
    /// MUTATION (recorded here, not left in the tree; Card 600 SC-002): in `Context::submit_cached`,
    /// change `bind_group: &d.bind_group` to `bind_group: &dispatches[0].bind_group` (every dispatch
    /// reuses dispatch 0's bind group - "two dispatches with different buffers" sharing one bind
    /// group). Result: RED - every dispatch now writes dispatch 0's output buffer, so outputs 1..3
    /// are never written and read back as their initial zero fill instead of their own 13.0/15.0/
    /// 17.0 (observed: dispatch 1 `left: [0.0; 17]` vs `right: [13.0; 17]`). Reverted: GREEN.
    #[test]
    fn dense_decode_replay_with_distinct_buffers_matches_exact_expected_gpu() {
        let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
            return;
        };
        let (dispatches, outputs) = four_dispatches(&ctx);
        let refs: Vec<&CachedDispatch> = dispatches.iter().collect();
        ctx.submit_cached(&refs).unwrap();
        for (i, output) in outputs.iter().enumerate() {
            let expected = vec![(i as f32 + 1.0) + (i as f32 + 10.0); 17];
            assert_eq!(
                ctx.download_f32(output).unwrap(),
                expected,
                "dispatch {i} must read back its own buffers, not a neighbor's"
            );
        }
    }

    /// `Context::submit_cached`'s pre-Card-600 shape, kept only here (never reintroduced into
    /// `dispatch.rs`): collect the dispatch list into a fresh `Vec<EncodeItem>`, then hand
    /// `submit_encoded` a fresh iterator rewrapping that `Vec`'s borrowed fields (so this helper
    /// needs no `EncodeItem: Clone` impl in production code - `slice::Iter` is `Clone` for any
    /// element type). Everything downstream of that hand-off - the dispatches, buffers, pipelines,
    /// bind groups and wgpu calls `submit_encoded` itself makes - is identical to the fixed path.
    fn submit_cached_pre_fix_shape(
        ctx: &Context,
        dispatches: &[&CachedDispatch],
    ) -> Result<(), crate::RuntimeError> {
        let items: Vec<EncodeItem> = dispatches
            .iter()
            .map(|d| EncodeItem {
                pipeline: &d.pipeline,
                bind_group: &d.bind_group,
                groups: d.groups,
                label: &d.label,
            })
            .collect();
        ctx.submit_encoded(items.iter().map(|e| EncodeItem {
            pipeline: e.pipeline,
            bind_group: e.bind_group,
            groups: e.groups,
            label: e.label,
        }))
    }

    /// SC-003 - the production `submit_cached`/`submit_encoded` path allocates *less than* the
    /// pre-Card-600 shape that collected its dispatch list into a `Vec<EncodeItem>` per call (the
    /// name states this relative contract, not an absolute "zero allocations" claim: wgpu's own
    /// per-submit bookkeeping - command-buffer tracking, validation - allocates regardless of this
    /// card and is out of its scope, so an absolute zero is not a reachable or meaningful
    /// assertion). Isolating the one allocation this card removed from wgpu's footprint: the fixed
    /// path and [`submit_cached_pre_fix_shape`] drive the identical dispatches through the
    /// identical `submit_encoded`/wgpu calls, so any allocation delta between them comes only from
    /// whether a `Vec<EncodeItem>` is collected first. A handful of untimed warm-up calls first let
    /// wgpu's own internal pools reach steady state (SC-001's "load + warm-up, then measure"
    /// shape), so that per-call wgpu allocation count is stable across both measured windows.
    ///
    /// MUTATION: this test *is* the SC-003 regression check - `submit_cached_pre_fix_shape` already
    /// reproduces the named mutation ("collect the dispatch list into a fresh Vec per step") without
    /// touching production code. With the mutation also applied to the production `submit_cached`
    /// (making both paths identical), the measured delta falls far under the `REPS / 2` threshold -
    /// observed fixed=2624, pre_fix=2628, delta=4 - RED. Reverted (the shapes differ by exactly one
    /// `Vec<EncodeItem>` collect per call): the delta reads a clear double-digit positive count
    /// every observed run (observed 16-24 in earlier, smaller-`REPS` runs; never anywhere near 0) -
    /// GREEN. The threshold asks only for "clearly positive, not noise-sized", not an exact count:
    /// wgpu's own per-call bookkeeping jitters by a few units run to run under box load, independent
    /// of this card's change, so an exact-count assertion is not a reliable gate. This relative
    /// comparison cannot by itself detect a *different* new per-dispatch allocation introduced
    /// elsewhere on the production path (review F4) - SC-001/SC-002 cover the rest of this card's
    /// surface.
    #[test]
    fn dense_decode_replay_allocates_less_than_the_pre_fix_shape_gpu() {
        let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
            return;
        };
        let (dispatches, _outputs) = four_dispatches(&ctx);
        let refs: Vec<&CachedDispatch> = dispatches.iter().collect();
        for _ in 0..8 {
            ctx.submit_cached(&refs).unwrap();
            submit_cached_pre_fix_shape(&ctx, &refs).unwrap();
        }
        const REPS: usize = 64;
        let [fixed_allocs, pre_fix_allocs] = super::alloc_counting::paired_allocation_counts(
            || ctx.submit_cached(&refs).unwrap(),
            || submit_cached_pre_fix_shape(&ctx, &refs).unwrap(),
            REPS,
        );
        let delta = pre_fix_allocs.saturating_sub(fixed_allocs);
        assert!(
            delta >= REPS / 2,
            "the pre-fix shape's only difference from the fixed `submit_cached` is one \
             Vec<EncodeItem> collected per call, so over {REPS} calls each it must cost clearly \
             more allocations, not merely as many or fewer (fixed={fixed_allocs}, \
             pre_fix={pre_fix_allocs}, delta={delta})"
        );
    }
}

/// Thread-local allocation counter installed as the test binary's global allocator (mirrors
/// `poot-eval`'s `src/tests/allocation.rs`; a separate test binary, so there is no conflict with
/// that crate's own `#[global_allocator]`). `#[cfg(test)]`-gated, so it is absent from the library
/// any production or dependent crate links against.
#[cfg(test)]
mod alloc_counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static RECORDING: Cell<bool> = const { Cell::new(false) };
        /// Which of [`paired_allocation_counts`]'s two buckets the next allocator call should land
        /// in - set right before each of the two interleaved closures runs, so position-dependent
        /// allocator/driver warm-up noise (e.g. a lazily-grown cache that only resizes on, say,
        /// every 8th cumulative call) lands on either bucket about equally instead of skewing
        /// whichever bucket happens to run as the second, later block of calls.
        static BUCKET: Cell<usize> = const { Cell::new(0) };
        static COUNTS: Cell<[usize; 2]> = const { Cell::new([0, 0]) };
    }

    struct CountingAllocator;

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    fn note_allocation() {
        if RECORDING.try_with(Cell::get).unwrap_or(false) {
            let bucket = BUCKET.try_with(Cell::get).unwrap_or(0);
            let _ = COUNTS.try_with(|counts| {
                let mut c = counts.get();
                c[bucket] += 1;
                counts.set(c);
            });
        }
    }

    // SAFETY: every method forwards to `System` with the caller's arguments unchanged and only
    // counts the call, so `System`'s allocator contract carries over.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            note_allocation();
            // SAFETY: forwarded unchanged under the caller's `GlobalAlloc::alloc` contract.
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            note_allocation();
            // SAFETY: forwarded unchanged under the caller's `GlobalAlloc::alloc_zeroed` contract.
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            // SAFETY: forwarded unchanged under the caller's `GlobalAlloc::dealloc` contract.
            unsafe { System.dealloc(pointer, layout) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            note_allocation();
            // SAFETY: forwarded unchanged under the caller's `GlobalAlloc::realloc` contract.
            unsafe { System.realloc(pointer, layout, new_size) }
        }
    }

    /// Run `a` then `b`, `reps` times each, interleaved (a, b, a, b, ...), and return each one's
    /// own allocation count. Interleaving - rather than running all of `a` then all of `b` - keeps
    /// any allocator/driver warm-up noise that depends only on cumulative call count from landing
    /// disproportionately on whichever closure happens to run second.
    pub(super) fn paired_allocation_counts(
        mut a: impl FnMut(),
        mut b: impl FnMut(),
        reps: usize,
    ) -> [usize; 2] {
        COUNTS.with(|counts| counts.set([0, 0]));
        RECORDING.with(|recording| recording.set(true));
        for _ in 0..reps {
            BUCKET.with(|bucket| bucket.set(0));
            a();
            BUCKET.with(|bucket| bucket.set(1));
            b();
        }
        RECORDING.with(|recording| recording.set(false));
        COUNTS.with(Cell::get)
    }
}
