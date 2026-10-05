use crate::replay::replay_graph_chunk_size;
use crate::*;

impl RocmContext {
    /// Initialize HSA, pick the first GPU agent, walk its memory pools, and create one cooperative GWS
    /// queue at max size (system-memory ring). Single-threaded; the context is `!Send`.
    pub fn new() -> Result<Self, RocmError> {
        Self::new_with_options(RocmContextOptions::default())
    }

    /// [`Self::new`] with an explicit [`RocmContextOptions`] (Card 548): the one
    /// construction-time seam for the bounded-wait ceiling, never an environment read.
    pub fn new_with_options(options: RocmContextOptions) -> Result<Self, RocmError> {
        Self::new_inner(QueueCreationPlan::CooperativeGws, options).map_err(require_gpu_check)
    }

    /// Receipt-only constructor: use ROCr's `HSA_ALLOCATE_QUEUE_DEV_MEM=1` device-memory packet ring,
    /// create an ordinary MULTI queue, and retain a `PcieDeviceMemory` publication contract after
    /// proving a non-APU agent and a CPU-to-GPU PCIe pool link.
    ///
    /// The process must be launched with `HSA_ALLOCATE_QUEUE_DEV_MEM=1` (ROCr reads it once at load);
    /// without it construction fails with [`RocmError::QueueRingProvenanceConflict`]. The library does
    /// not set it: mutating the environment can race other threads. Does not change production
    /// cooperative-GWS construction or Card 276 admission policy.
    pub fn new_for_pcie_device_ring_publication_receipt() -> Result<Self, RocmError> {
        Self::new_inner(
            QueueCreationPlan::PcieDeviceMemoryRingReceipt,
            RocmContextOptions::default(),
        )
        .map_err(require_gpu_check)
    }

    pub(crate) fn new_inner(
        plan: QueueCreationPlan,
        options: RocmContextOptions,
    ) -> Result<Self, RocmError> {
        let override_now = device_ring_override_present(|name| std::env::var_os(name));
        let override_at_load = *DEVICE_RING_OVERRIDE_AT_FIRST_HSA_LOAD.get_or_init(|| override_now);
        validate_queue_plan_provenance(plan, override_at_load, override_now)?;
        let hsa = Hsa::load()?;

        // hsa_init bumps the runtime's refcount; `RocmContext`'s Drop pairs it with shut_down on success,
        // and `shutdown_guard` below on every early return.
        // SAFETY: `hsa` was just loaded and resolved; `hsa_init` takes no arguments.
        unsafe {
            check((hsa.funcs.hsa_init)())?;
        }
        info!("hsa_init OK");
        let mut shutdown_guard = HsaShutdownGuard {
            shut_down: hsa.funcs.hsa_shut_down,
            armed: true,
        };

        // Pass the funcs pointer through the iteration callback data so the `unsafe extern "C"` callback
        // can use them without a thread-local.
        let picker = GpuPicker {
            funcs: &hsa.funcs,
            picked: None,
        };
        let mut picker = picker;
        unsafe extern "C" fn gpu_pick_cb(agent: hsa_agent_t, data: *mut c_void) -> hsa_status_t {
            // SAFETY: `data` is the `&mut picker` passed to `hsa_iterate_agents` below, which calls back
            // synchronously while `picker` is otherwise unused; its table belongs to the initialized
            // runtime.
            unsafe {
                let picker = &mut *(data as *mut GpuPicker<'_>);
                picker.visit(agent)
            }
        }
        // SAFETY: `gpu_pick_cb` matches the callback type and `picker` outlives the synchronous
        // iteration.
        let iterate_status = unsafe {
            (hsa.funcs.hsa_iterate_agents)(Some(gpu_pick_cb), &mut picker as *mut _ as *mut c_void)
        };
        check(iterate_status)?;
        let gpu_agent = picker.picked.ok_or(RocmError::NoGpuAgent)?;

        // Log the agent's name + vendor for diagnostics. Each discovery query below meets the
        // `discovery` module contract: `hsa` is loaded and initialized.
        // SAFETY: discovery contract (above).
        let name = unsafe { agent_name(&hsa.funcs, gpu_agent)? };
        // SAFETY: discovery contract (above).
        let vendor = unsafe { agent_vendor(&hsa.funcs, gpu_agent)? };
        // SAFETY: discovery contract (above).
        let product = unsafe { amd_agent_product_name(&hsa.funcs, gpu_agent) };
        // SAFETY: discovery contract (above).
        let wavefront = unsafe { agent_wavefront_size(&hsa.funcs, gpu_agent)? };
        // SAFETY: discovery contract (above).
        let coop_q = unsafe { amd_cooperative_queues(&hsa.funcs, gpu_agent) };
        // SAFETY: discovery contract (above).
        let isa = unsafe { agent_isa_name(&hsa.funcs, gpu_agent)? };
        info!(
            "picked GPU agent: name={name:?} vendor={vendor:?} product={product:?} \
             wavefront={wavefront} coop_queues={coop_q:?} isa={isa:?}"
        );

        // Walk the GPU agent's memory pools: a coarse device pool (output buffers) and a fine system pool
        // (kernarg; on Strix Halo also zero-copy H2D/D2H).
        let mut coarse_pool: Option<hsa_amd_memory_pool_t> = None;
        let mut fine_pool: Option<hsa_amd_memory_pool_t> = None;
        let mut pool_state = PoolState {
            coarse: &mut coarse_pool,
            fine: &mut fine_pool,
            funcs: &hsa.funcs,
        };
        unsafe extern "C" fn pool_cb(
            pool: hsa_amd_memory_pool_t,
            data: *mut c_void,
        ) -> hsa_status_t {
            // SAFETY: `data` is the `&mut pool_state` passed below, used only by this synchronous
            // callback; its table belongs to the initialized runtime.
            unsafe {
                let state = &mut *(data as *mut PoolState<'_>);
                state.visit(pool)
            }
        }
        // SAFETY: `pool_cb` matches the callback type and `pool_state` outlives the synchronous
        // iteration.
        let pool_iter_status = unsafe {
            (hsa.funcs.hsa_amd_agent_iterate_memory_pools)(
                gpu_agent,
                Some(pool_cb),
                &mut pool_state as *mut _ as *mut c_void,
            )
        };
        check(pool_iter_status)?;
        let gpu_coarse_pool = coarse_pool.ok_or(RocmError::AgentEnum(
            "no COARSE_GRAINED GPU device pool (broken ROCm install?)",
        ))?;
        let fine_pool = fine_pool.ok_or(RocmError::AgentEnum(
            "no FINE_GRAINED system pool (broken ROCm install?)",
        ))?;

        // Card 125 (FR-004/SC-004): decide the upload strategy for `fine_pool`.
        let mut fine_pool_location: u32 = 0;
        // SAFETY: `HSA_AMD_MEMORY_POOL_INFO_LOCATION` writes a 4-byte enum into the local.
        unsafe {
            check((hsa.funcs.hsa_amd_memory_pool_get_info)(
                fine_pool,
                HSA_AMD_MEMORY_POOL_INFO_LOCATION,
                &mut fine_pool_location as *mut _ as *mut c_void,
            ))?;
        }
        let mut fine_pool_flags: u32 = 0;
        // SAFETY: `HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS` writes a `uint32_t` into the local.
        unsafe {
            check((hsa.funcs.hsa_amd_memory_pool_get_info)(
                fine_pool,
                HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                &mut fine_pool_flags as *mut _ as *mut c_void,
            ))?;
        }
        let upload_strategy = upload_strategy_for_pool(fine_pool_location, fine_pool_flags);
        info!(?upload_strategy, "resolved upload strategy for fine_pool");

        let mut max_size: u32 = 0;
        // SAFETY: `HSA_AGENT_INFO_QUEUE_MAX_SIZE` writes a `uint32_t` into the local.
        unsafe {
            check((hsa.funcs.hsa_agent_get_info)(
                gpu_agent,
                HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                &mut max_size as *mut _ as *mut c_void,
            ))?;
        }

        let queue_ring_provenance = plan.provenance();
        let (ring_memory, host_link, mut device_ring_receipt) = match plan {
            QueueCreationPlan::CooperativeGws => {
                // ROCr builds the cooperative GWS queue through CreateInterceptibleQueue with the explicit
                // HSA_AMD_QUEUE_CREATE_SYSTEM_MEM allocation flag.
                let (ring_memory, host_link) = cooperative_gws_publication_facts();
                (ring_memory, host_link, None)
            }
            #[cfg(test)]
            QueueCreationPlan::RecordedReplay => {
                let (ring_memory, host_link) = ordinary_system_publication_facts();
                (ring_memory, host_link, None)
            }
            QueueCreationPlan::PcieDeviceMemoryRingReceipt => {
                // SAFETY: discovery contract (above).
                if unsafe { amd_agent_is_apu(&hsa.funcs, gpu_agent)? } {
                    return Err(RocmError::ApuNotDiscrete);
                }
                // SAFETY: discovery contract (above).
                let cpu_agent =
                    unsafe { first_cpu_agent(&hsa.funcs)? }.ok_or(RocmError::NoCpuAgent)?;
                // SAFETY: discovery contract (above).
                let (num_hops, link_types) =
                    unsafe { cpu_to_pool_link_types(&hsa.funcs, cpu_agent, gpu_coarse_pool)? };
                if !link_types.contains(&HSA_AMD_LINK_INFO_TYPE_PCIE) {
                    return Err(RocmError::HostLinkNotPcie {
                        num_hops,
                        link_types,
                    });
                }
                // ROCr's HSA_ALLOCATE_QUEUE_DEV_MEM=1 (set before hsa_init) selects
                // HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF for ordinary MULTI/SINGLE creates.
                let ring_memory = Some(QueueRingMemory::DeviceMemory);
                let host_link = Some(QueueHostLink::Pcie);
                // SAFETY: discovery contract; BDFID and DOMAIN are `uint32_t` attributes.
                let bdfid =
                    unsafe { amd_agent_u32_info(&hsa.funcs, gpu_agent, HSA_AMD_AGENT_INFO_BDFID) };
                // SAFETY: as above.
                let pci_domain =
                    unsafe { amd_agent_u32_info(&hsa.funcs, gpu_agent, HSA_AMD_AGENT_INFO_DOMAIN) };
                let product_name = product.clone().unwrap_or_else(|| name.clone());
                let receipt = DeviceRingReceiptProvenance {
                    allocate_queue_dev_mem: std::env::var_os("HSA_ALLOCATE_QUEUE_DEV_MEM")
                        .is_some_and(|v| v == "1"),
                    queue_api: "hsa_queue_create(MULTI) + HSA_ALLOCATE_QUEUE_DEV_MEM=1",
                    queue_type_requested: HSA_QUEUE_TYPE_MULTI,
                    queue_type_returned: 0,
                    queue_features: 0,
                    queue_size: 0,
                    agent_name: name.clone(),
                    product_name,
                    isa_name: isa.clone(),
                    bdfid,
                    pci_domain,
                    num_link_hops: num_hops,
                    link_types,
                    contract: QueuePublicationContract::HostCoherent,
                };
                (ring_memory, host_link, Some(receipt))
            }
        };

        let queue_publication = publication::QueuePublication::validate(
            ring_memory,
            host_link,
            QueueHostTarget::current(),
        )?;
        let request = queue_request(plan, max_size);
        let mut queue: *mut hsa_queue_t = std::ptr::null_mut();
        // SAFETY: `gpu_agent` belongs to the initialized runtime; no error callback or data; `queue`
        // is a valid out-pointer.
        let create_status = unsafe {
            (hsa.funcs.hsa_queue_create)(
                gpu_agent,
                request.size,
                request.queue_type,
                None,
                std::ptr::null_mut(),
                0,
                0,
                &mut queue,
            )
        };
        if create_status != bindings::HSA_STATUS_SUCCESS {
            // SAFETY: `acquired` is a non-null queue the failed create still returned, destroyed once.
            let _ = destroy_queue_reference(&mut queue, |acquired| unsafe {
                (hsa.funcs.hsa_queue_destroy)(acquired)
            });
            if matches!(plan, QueueCreationPlan::PcieDeviceMemoryRingReceipt) {
                return Err(RocmError::DeviceRingQueueCreate(HsaStatus {
                    code: create_status,
                }));
            }
            check(create_status)?;
        }
        if queue.is_null() {
            return Err(RocmError::QueueContract(
                "HSA returned success with a null queue".to_string(),
            ));
        }
        // SAFETY: `queue` is the non-null queue just created; `QueueLayout` mirrors `hsa_queue_t`, whose
        // fields the runtime does not write after creation.
        let q_layout = unsafe { &*(queue as *const QueueLayout) };
        let actual = QueueProperties::from(q_layout);
        if let Err(error) = validate_created_queue(request, actual) {
            // SAFETY: the queue was just created and never used, so it is destroyed once with no
            // packet in flight.
            let _ = destroy_queue_reference(&mut queue, |acquired| unsafe {
                (hsa.funcs.hsa_queue_destroy)(acquired)
            });
            return Err(error);
        }
        if let Some(receipt) = device_ring_receipt.as_mut() {
            receipt.queue_type_returned = q_layout.r#type;
            receipt.queue_features = q_layout.features;
            receipt.queue_size = q_layout.size;
            receipt.contract = queue_publication.contract();
        }
        info!(
            publication = ?queue_publication.contract(),
            plan = ?plan,
            role = ?plan.role(),
            provenance = ?queue_ring_provenance,
            requested_type = request.queue_type,
            requested_size = request.size,
            returned_type = actual.queue_type,
            returned_features = actual.features,
            returned_size = actual.size,
            replay_batch_capacity = replay_graph_chunk_size(actual.size as u64),
            "created HSA queue"
        );

        // SAFETY: discovery contract (above).
        let vram_total_bytes = unsafe { amd_pool_size(&hsa.funcs, gpu_coarse_pool) }.unwrap_or(0);
        // SAFETY: discovery contract (above); the attribute's value is a `uint32_t`.
        let compute_units = unsafe {
            amd_agent_u32_info(&hsa.funcs, gpu_agent, HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT)
        };

        // From here the shared runtime owner's `Drop` owns `hsa_shut_down`; disarm the guard so it does
        // not also fire.
        shutdown_guard.armed = false;

        Ok(RocmContext {
            hsa: Arc::new(HsaRuntimeOwner {
                hsa,
                memory: poot_runtime_common::MemoryCounters::new(),
            }),
            gpu_agent,
            isa_name: isa,
            wavefront,
            fine_pool,
            coarse_pool: gpu_coarse_pool,
            upload_strategy,
            queue,
            queue_role: plan.role(),
            queue_ring_provenance,
            queue_publication,
            device_ring_receipt,
            vram_total_bytes,
            compute_units,
            timeouts: TimeoutLeaks::default(),
            wait_timeout: options.wait_timeout,
        })
    }

    /// Device-wide VRAM in use (bytes): `HSA_AMD_MEMORY_POOL_INFO_SIZE` (cached total) minus a live
    /// `hsa_agent_get_info(HSA_AMD_AGENT_INFO_MEMORY_AVAIL)` read. The ROCm/HSA analogue of
    /// `poot-runtime::Context::vram_used_bytes` (Vulkan `VK_EXT_memory_budget`) and
    /// `PtxContext::vram_used_bytes` (CUDA `cuMemGetInfo`) (card 030). Engine-queried, no sysfs or
    /// rocm-smi. HSA has no "used" attribute, so it is derived from the two queries. `None` if the
    /// driver call fails.
    ///
    /// Card 268 measurement: unlike Vulkan's `VK_EXT_memory_budget` (which accounts for other
    /// processes), HSA's `MEMORY_AVAIL` covers only memory allocated through HSA/KFD's own pool
    /// accounting. With ~79 GiB held by co-resident non-HSA GPU clients (`free -h` available ~45 GiB) it
    /// still reported ~94.6 GiB "available". Do not use it as a system-wide free-memory signal on a
    /// shared unified-memory box.
    pub fn vram_used_bytes(&self) -> Option<u64> {
        if self.vram_total_bytes == 0 {
            return None; // the one-time SIZE query at construction failed - no total to subtract from.
        }
        // SAFETY: discovery contract: the context keeps its runtime initialized.
        let avail = unsafe { amd_agent_memory_avail(&self.hsa.funcs, self.gpu_agent) }.ok()?;
        Some(self.vram_total_bytes.saturating_sub(avail))
    }

    /// Total memory of the GPU agent's device pool (bytes): the coarse-grained pool's
    /// `HSA_AMD_MEMORY_POOL_INFO_SIZE`, cached at construction (capacity is fixed for the process
    /// lifetime). `free = vram_budget_bytes - vram_used_bytes` is the headroom. `None` if the one-time
    /// SIZE query failed.
    ///
    /// Card 268: `free` is not a usable live-headroom signal for VRAM auto-fit on a shared unified-memory
    /// box, because `vram_used_bytes`'s `MEMORY_AVAIL` does not see co-resident non-HSA GPU clients (see
    /// that method). No auto-fit call site should treat this pair as a contention signal.
    pub fn vram_budget_bytes(&self) -> Option<u64> {
        if self.vram_total_bytes == 0 {
            None
        } else {
            Some(self.vram_total_bytes)
        }
    }

    /// Card 125 (FR-004/SC-004): a device DMA transfer (`hsa_amd_memory_async_copy` + completion
    /// signal) for [`Self::upload_strategy`] == [`UploadStrategy::Dma`]. `size` is bytes; `dst`/`src`
    /// are a pool allocation and the agent address of `staging`. The copy starts once every signal in
    /// `after` is below 1.
    ///
    /// Card 602: the copy takes the staging buffer and hands it back once the copy is proven
    /// complete, so the caller reads it only then. When the wait times out the staging buffer stays
    /// pinned and allocated on the leak list.
    ///
    /// # Safety
    ///
    /// One of `dst`/`src` is `staging`'s agent address and the other a live pool allocation of this
    /// runtime; both must be valid for `size` bytes (`staging` holds at least `size`).
    pub(crate) unsafe fn dma_copy(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        staging: StagingBuffer,
        after: &[bindings::hsa_signal_t],
    ) -> Result<StagingBuffer, RocmError> {
        self.ensure_not_poisoned("dma_copy")?;
        let mut completion = bindings::hsa_signal_t { handle: 0 };
        // SAFETY: zero consumers with a null list is allowed; `completion` is a valid out-pointer.
        unsafe {
            check((self.hsa.funcs.hsa_signal_create)(
                1,
                0,
                std::ptr::null(),
                &mut completion,
            ))?;
        }
        let after_count = u32::try_from(after.len()).expect("DMA dependency count fits in u32");
        // SAFETY: both ranges are valid for `size` bytes (function contract) and stay alive until the
        // copy is proven complete: the staging buffer moves into the wait below, which leaks it on
        // timeout, and a timeout poisons the runtime so no pool allocation is freed. `after` holds
        // `after_count` live signals and outlives the call.
        let status = unsafe {
            (self.hsa.funcs.hsa_amd_memory_async_copy)(
                dst,
                self.gpu_agent,
                src,
                self.gpu_agent,
                size,
                after_count,
                if after.is_empty() {
                    std::ptr::null()
                } else {
                    after.as_ptr()
                },
                completion,
            )
        };
        if status != bindings::HSA_STATUS_SUCCESS {
            // SAFETY: the copy was not submitted, so nothing waits on `completion`; destroyed once.
            unsafe {
                let _ = (self.hsa.funcs.hsa_signal_destroy)(completion);
            }
            return Err(RocmError::Driver(HsaStatus { code: status }));
        }
        let held = self.wait_completion_bounded(
            completion,
            "dma_copy",
            Held {
                staging: Some(staging),
            },
        )?;
        Ok(held
            .staging
            .expect("a completed wait hands back the staging buffer it was given"))
    }

    /// Card 602: DMA `size` host bytes at `src` into device pointer `dst` through a runtime-owned
    /// staging buffer. The host copies `src` into staging before the copy is submitted, so a timeout
    /// leaves no device reference to the caller's memory.
    ///
    /// # Safety
    /// `dst` must point to a live allocation of at least `size` bytes; `src` must be valid to read for
    /// `size` bytes.
    pub(crate) unsafe fn dma_upload(
        &self,
        src: *const u8,
        dst: *mut c_void,
        size: usize,
    ) -> Result<(), RocmError> {
        let mut staging = self.dma_staging(size)?;
        // SAFETY: `src` is readable for `size` bytes (function contract) and `staging` holds `size`
        // freshly allocated bytes, so the ranges do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(src, staging.host_mut_ptr(), size);
        }
        let staged = staging.agent_ptr();
        // SAFETY: `staged` is `staging`'s agent address and `dst` a live allocation of `size` bytes
        // (function contract).
        unsafe { self.dma_copy(dst, staged, size, staging, &[]) }?;
        Ok(())
    }

    /// Card 602: DMA `size` device bytes at `src` into host pointer `dst` through a runtime-owned
    /// staging buffer, once every signal in `after` is below 1. The engine writes only the staging
    /// buffer; the host copies it into `dst` after the copy is proven complete, so a timeout leaves the
    /// caller's memory untouched and free to drop or reuse.
    ///
    /// # Safety
    /// `src` must point to a live allocation of at least `size` bytes; `dst` must be valid to write
    /// for `size` bytes.
    pub(crate) unsafe fn dma_download(
        &self,
        src: *const c_void,
        dst: *mut u8,
        size: usize,
        after: &[bindings::hsa_signal_t],
    ) -> Result<(), RocmError> {
        let staging = self.dma_staging(size)?;
        let staged = staging.agent_ptr();
        // SAFETY: `staged` is `staging`'s agent address and `src` a live allocation of `size` bytes
        // (function contract).
        let staging = unsafe { self.dma_copy(staged, src, size, staging, after) }?;
        debug_assert_eq!(staging.len(), size);
        // SAFETY: the copy completed, `staging` holds `size` bytes, and `dst` is writable for `size`
        // bytes (function contract) and distinct from runtime-owned staging memory.
        unsafe {
            std::ptr::copy_nonoverlapping(staging.host_ptr(), dst, size);
        }
        Ok(())
    }

    /// Card 125 (FR-004/SC-004): write `size` bytes from host pointer `src` to device pointer `dst`,
    /// branching on [`Self::upload_strategy`]: `ptr::copy_nonoverlapping` when `fine_pool` is
    /// host-visible (`HostWrite`, APU/HMM), or a staged DMA ([`Self::dma_upload`]) when it is not
    /// (`Dma`, discrete VRAM).
    ///
    /// # Safety
    /// `dst` must point to a live allocation of at least `size` bytes from `self.fine_pool`
    /// (or another pool `self.gpu_agent` can access); `src` must be valid to read for `size`
    /// bytes.
    pub(crate) unsafe fn upload_bytes_into(
        &self,
        src: *const u8,
        dst: *mut c_void,
        size: usize,
    ) -> Result<(), RocmError> {
        if size == 0 {
            // A zero-size `hsa_amd_memory_lock`/`hsa_amd_memory_async_copy` is undocumented, and a zero-byte
            // transfer has nothing to do.
            return Ok(());
        }
        match self.upload_strategy {
            UploadStrategy::HostWrite => {
                // SAFETY: `HostWrite` means the pool is host-visible, so `dst` is a host address valid
                // for `size` bytes, as is `src` (function contract); host and pool memory do not overlap.
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst as *mut u8, size);
                }
                Ok(())
            }
            // SAFETY: the function contract is `dma_upload`'s.
            UploadStrategy::Dma => unsafe { self.dma_upload(src, dst, size) },
        }
    }

    /// Card 125 (FR-004/SC-004): download mirror of [`Self::upload_bytes_into`]: read `size` bytes from
    /// device pointer `src` into host pointer `dst`.
    ///
    /// # Safety
    /// `src` must point to a live allocation of at least `size` bytes; `dst` must be valid to
    /// write for `size` bytes.
    pub(crate) unsafe fn download_bytes_from(
        &self,
        src: *const c_void,
        dst: *mut u8,
        size: usize,
    ) -> Result<(), RocmError> {
        if size == 0 {
            return Ok(());
        }
        match self.upload_strategy {
            UploadStrategy::HostWrite => {
                // SAFETY: `HostWrite` means `src` is a host-visible pool address readable for `size`
                // bytes, and `dst` is writable for `size` bytes (function contract); they do not overlap.
                unsafe {
                    std::ptr::copy_nonoverlapping(src as *const u8, dst, size);
                }
                Ok(())
            }
            // SAFETY: the function contract is `dma_download`'s.
            UploadStrategy::Dma => unsafe { self.dma_download(src, dst, size, &[]) },
        }
    }

    /// Card 547a: [`Self::upload_bytes_into`] for a fine-pool buffer, or an unconditional staged DMA
    /// ([`Self::dma_upload`]) for a coarse-pool one - the coarse-grained pool is never assumed
    /// host-mappable (the gfx1151 risk the card names), so a coarse write always bridges through a
    /// pinned host staging buffer and a device-to-device copy, regardless of `self.upload_strategy`
    /// (which describes `fine_pool` only).
    ///
    /// # Safety
    /// Same contract as [`Self::upload_bytes_into`]: `dst` must point to a live allocation of at least
    /// `size` bytes in the pool `pool` names; `src` must be valid to read for `size` bytes.
    pub(crate) unsafe fn upload_bytes_role_aware(
        &self,
        pool: Pool,
        src: *const u8,
        dst: *mut c_void,
        size: usize,
    ) -> Result<(), RocmError> {
        match pool {
            // SAFETY: the function contract matches `upload_bytes_into`'s.
            Pool::Fine => unsafe { self.upload_bytes_into(src, dst, size) },
            // SAFETY: the function contract matches `dma_upload`'s.
            Pool::Coarse => unsafe { self.dma_upload(src, dst, size) },
        }
    }

    /// Download mirror of [`Self::upload_bytes_role_aware`].
    ///
    /// # Safety
    /// Same contract as [`Self::download_bytes_from`]: `src` must point to a live allocation of at
    /// least `size` bytes in the pool `pool` names; `dst` must be valid to write for `size` bytes.
    pub(crate) unsafe fn download_bytes_role_aware(
        &self,
        pool: Pool,
        src: *const c_void,
        dst: *mut u8,
        size: usize,
    ) -> Result<(), RocmError> {
        match pool {
            // SAFETY: the function contract matches `download_bytes_from`'s.
            Pool::Fine => unsafe { self.download_bytes_from(src, dst, size) },
            // SAFETY: the function contract matches `dma_download`'s.
            Pool::Coarse => unsafe { self.dma_download(src, dst, size, &[]) },
        }
    }

    pub(crate) fn allocate_uninit(
        &self,
        op: &'static str,
        elems: usize,
        storage: BufferStorage,
        role: BufferRole,
    ) -> Result<RocmBuffer, RocmError> {
        let (elem_count, bytes) = checked_layout(op, elems, storage)?;
        let pool = Pool::for_role(role);
        let native_pool = self.native_pool(pool);
        let ptr = self.pool_alloc(native_pool, bytes)?;
        Ok(self.own_buffer(ptr, elem_count, bytes, storage, role, pool))
    }

    pub(crate) fn allocate_zeroed(
        &self,
        op: &'static str,
        elems: usize,
        storage: BufferStorage,
        role: BufferRole,
    ) -> Result<RocmBuffer, RocmError> {
        let buf = self.allocate_uninit(op, elems, storage, role)?;
        let zeros = vec![0u8; buf.byte_capacity()];
        // SAFETY: `buf` is a fresh allocation of exactly `byte_capacity()` bytes, and `zeros` holds
        // that many readable bytes.
        unsafe {
            self.upload_bytes_role_aware(
                buf.pool(),
                zeros.as_ptr(),
                buf.alloc.as_ptr(),
                buf.byte_capacity(),
            )?;
        }
        Ok(buf)
    }

    /// Allocate a zeroed buffer of `elems` native elements of `storage`, charged to `role` (Card 548:
    /// the one storage-generic allocation primitive `RocmDevice::allocate` calls through the
    /// `poot_executor::Device` contract - every typed `allocate_*` below is a thin specialization of
    /// this for a fixed storage). Parameter order matches [`poot_executor::Device::allocate`]
    /// (`role, storage, elems`), so a `Device` impl forwards with no reordering. See
    /// [`Self::allocate_f32`] for the pool/write-path rule `Pool::for_role(role)` decides.
    pub fn allocate(
        &self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<RocmBuffer, RocmError> {
        self.allocate_zeroed("allocate", elems, storage, role)
    }

    /// Allocate a zeroed buffer of `elems` f32, charged to `role` (Card 547a review F1/F4: the caller
    /// states the role; this no longer hardcodes `Activation`). `Pool::for_role(role)` decides which
    /// native pool backs it: `Weight`/`Activation`/`State` resolve to the coarse-grained device pool,
    /// so a later write bridges through a staged DMA ([`Self::dma_upload`]); `Input`/`Output`/`Meta`/
    /// `Staging` resolve to the fine-grained system pool, so a later write on Strix Halo (APU, HMM) is
    /// a plain host `ptr::copy_nonoverlapping`. Callers on the hot per-token path (`poot-rocm-gpu`'s
    /// per-step Token/Pos/Mask/SlotMap/GdnSlotMap/TokenEmbed slots) must pass `BufferRole::Input` to
    /// get that plain write; passing `Activation` there pays a staged-DMA per token instead.
    pub fn allocate_f32(&self, elems: usize, role: BufferRole) -> Result<RocmBuffer, RocmError> {
        self.allocate_zeroed("allocate_f32", elems, BufferStorage::f32(), role)
    }

    /// Allocate an uninitialized f32 buffer of `elems`, charged to `role` (Card 547a review F1/F4:
    /// caller-stated role). Same as [`Self::allocate_f32`] without the zeroing pass; only for callers
    /// that immediately overwrite the whole buffer (e.g. [`Self::upload_f32`]). On a 35B ROCm preload
    /// the zeroing pass alone costs tens of GB of single-threaded writes to slow fine-grained GTT
    /// memory.
    pub fn allocate_f32_uninit(
        &self,
        elems: usize,
        role: BufferRole,
    ) -> Result<RocmBuffer, RocmError> {
        self.allocate_uninit("allocate_f32_uninit", elems, BufferStorage::f32(), role)
    }

    /// Upload `data` into a fresh buffer charged to `role` (Card 547a review F1/F4: caller-stated role;
    /// production weight uploads, `upload_dense_const_lane`, pass `BufferRole::Weight`). Same pool +
    /// zero-copy semantics as [`Self::allocate_f32_uninit`].
    pub fn upload_f32(&self, data: &[f32], role: BufferRole) -> Result<RocmBuffer, RocmError> {
        let buf = self.allocate_f32_uninit(data.len(), role)?;
        // SAFETY: `buf` is a fresh allocation of exactly `size_of_val(data)` bytes, and `data` is
        // readable for that many.
        unsafe {
            self.upload_bytes_role_aware(
                buf.pool(),
                data.as_ptr() as *const u8,
                buf.alloc.as_ptr(),
                std::mem::size_of_val(data),
            )?;
        }
        Ok(buf)
    }

    /// Upload `data` (i32 slice) into a fresh native-I32 buffer charged to `role` (Card 547a F1/F4: caller-stated role; metadata buffers, e.g. ComputeMeta's dims, pass `BufferRole::Meta`),
    /// copying the i32 bytes directly. The bit pattern is identical to u32 for small non-negative
    /// values.
    pub fn upload_i32(&self, data: &[i32], role: BufferRole) -> Result<RocmBuffer, RocmError> {
        let buf = self.allocate_uninit("upload_i32", data.len(), BufferStorage::i32(), role)?;
        // SAFETY: `buf` is a fresh allocation of exactly `size_of_val(data)` bytes, and `data` is
        // readable for that many.
        unsafe {
            self.upload_bytes_role_aware(
                buf.pool(),
                data.as_ptr() as *const u8,
                buf.alloc.as_ptr(),
                std::mem::size_of_val(data),
            )?;
        }
        Ok(buf)
    }

    /// Upload raw bf16 bytes unchanged into a native two-byte-per-element buffer charged to `role`
    /// (Card 547a review F1/F4: caller-stated role).
    pub fn upload_bf16_bytes(
        &self,
        bytes: &[u8],
        role: BufferRole,
    ) -> Result<RocmBuffer, RocmError> {
        if !bytes
            .len()
            .is_multiple_of(poot_target::ElementKind::Bf16.byte_width())
        {
            return Err(RocmError::InvalidByteLength {
                op: "upload_bf16_bytes",
                bytes: bytes.len(),
                element_bytes: poot_target::ElementKind::Bf16.byte_width(),
            });
        }
        let elements = bytes.len() / poot_target::ElementKind::Bf16.byte_width();
        let buf =
            self.allocate_uninit("upload_bf16_bytes", elements, BufferStorage::bf16(), role)?;
        // SAFETY: `buf` is a fresh allocation of exactly `bytes.len()` bytes (whole elements).
        unsafe {
            self.upload_bytes_role_aware(
                buf.pool(),
                bytes.as_ptr(),
                buf.alloc.as_ptr(),
                bytes.len(),
            )?;
        }
        Ok(buf)
    }

    /// Overwrite an existing device buffer in place, through whichever pool it was allocated from
    /// (Card 547a: pool-aware, `buf.pool()`). On Strix Halo (HMM/SVM) a fine-pool buffer (`Input`/
    /// `Output`/`Meta`/`Staging`, Card 547a review F1/F4) gets the same host `ptr::copy_nonoverlapping`
    /// as before; a coarse-pool buffer (`Weight`/`Activation`/`State`) always bridges through a pinned
    /// host staging copy ([`Self::dma_upload`]). `poot-rocm-gpu`'s `decode_step` refreshes the Token,
    /// Pos, and Mask slot buffers between steps without re-allocating through this; those slots are
    /// allocated with `BufferRole::Input`, so this is the unchanged plain host write on the hot
    /// per-token path - a caller that allocates a per-step slot under `Activation` instead pays a
    /// staged DMA on every token (the regression this guards against).
    pub fn update_f32(&self, buf: &RocmBuffer, data: &[f32]) -> Result<(), RocmError> {
        checked_copy(
            buf,
            "update_f32",
            BufferStorage::f32(),
            0,
            data.len(),
            true,
            // SAFETY: `checked_copy` proved `data` covers the whole buffer, so `byte_len` bytes are
            // readable from `data` and writable at the start of the live allocation.
            |range| unsafe {
                self.upload_bytes_role_aware(
                    buf.pool(),
                    data.as_ptr() as *const u8,
                    buf.alloc.as_ptr(),
                    range.byte_len,
                )
            },
        )
    }

    /// Write raw bytes into an owned kernarg allocation, through whichever pool it was allocated from
    /// (Card 547a). The representation and byte range are validated before pointer arithmetic or
    /// host/DMA copy.
    pub fn write_raw_bytes(
        &self,
        buf: &RocmBuffer,
        byte_offset: usize,
        data: &[u8],
    ) -> Result<(), RocmError> {
        checked_copy(
            buf,
            "write_raw_bytes",
            BufferStorage::raw_bytes(),
            byte_offset,
            data.len(),
            false,
            // SAFETY: `checked_copy` proved `[byte_offset, byte_offset + byte_len)` lies inside the live
            // allocation and `byte_len` equals `data.len()`.
            |range| unsafe {
                let dst = (buf.alloc.as_ptr() as *mut u8).add(range.byte_offset);
                self.upload_bytes_role_aware(
                    buf.pool(),
                    data.as_ptr(),
                    dst as *mut c_void,
                    range.byte_len,
                )
            },
        )
    }

    /// Download the contents of `buf` into `out`, which must be sized to `buf.elem_count()`. Through
    /// whichever pool `buf` was allocated from (Card 547a). Synchronizes first so any queued kernel
    /// that wrote the buffer has completed.
    pub fn download_f32(&self, buf: &RocmBuffer, out: &mut [f32]) -> Result<(), RocmError> {
        checked_copy(
            buf,
            "download_f32",
            BufferStorage::f32(),
            0,
            out.len(),
            true,
            |range| {
                self.synchronize()?;
                // SAFETY: `checked_copy` proved `out` covers the whole live allocation, so `byte_len`
                // bytes are readable from it and writable into `out`; the queue is drained.
                unsafe {
                    self.download_bytes_role_aware(
                        buf.pool(),
                        buf.alloc.as_ptr() as *const c_void,
                        out.as_mut_ptr() as *mut u8,
                        range.byte_len,
                    )
                }
            },
        )
    }

    /// Download packed i32/u32 words without changing their bit representation. Through whichever
    /// pool `buf` was allocated from (Card 547a).
    pub fn download_i32(&self, buf: &RocmBuffer, out: &mut [i32]) -> Result<(), RocmError> {
        checked_copy(
            buf,
            "download_i32",
            BufferStorage::i32(),
            0,
            out.len(),
            true,
            |range| {
                self.synchronize()?;
                // SAFETY: `checked_copy` proved `out` covers the whole live allocation, so `byte_len`
                // bytes are readable from it and writable into `out`; the queue is drained.
                unsafe {
                    self.download_bytes_role_aware(
                        buf.pool(),
                        buf.alloc.as_ptr() as *const c_void,
                        out.as_mut_ptr() as *mut u8,
                        range.byte_len,
                    )
                }
            },
        )
    }

    /// Overwrite the first `bytes.len()` bytes of `buf`, through whichever pool it was allocated from
    /// (Card 547a: the storage-typed, dtype-agnostic write the executor contract's memory service
    /// needs, instead of one allocator/writer per dtype; mirrors `poot_runtime::Context::write_bytes`).
    pub fn write_bytes(&self, buf: &RocmBuffer, bytes: &[u8]) -> Result<(), RocmError> {
        let range = checked_byte_range("write_bytes", buf, bytes.len())?;
        // SAFETY: `range` proved `bytes.len()` lies inside `buf`'s live allocation.
        unsafe {
            self.upload_bytes_role_aware(
                buf.pool(),
                bytes.as_ptr(),
                buf.alloc.as_ptr(),
                range.byte_len,
            )
        }
    }

    /// Read the first `out.len()` bytes of `buf` into `out`, through whichever pool it was allocated
    /// from (Card 547a). Synchronizes first so any queued kernel that wrote the buffer has completed.
    pub fn read_bytes(&self, buf: &RocmBuffer, out: &mut [u8]) -> Result<(), RocmError> {
        let range = checked_byte_range("read_bytes", buf, out.len())?;
        self.synchronize()?;
        // SAFETY: `range` proved `out.len()` lies inside `buf`'s live allocation, and the queue is
        // drained.
        unsafe {
            self.download_bytes_role_aware(
                buf.pool(),
                buf.alloc.as_ptr() as *const c_void,
                out.as_mut_ptr(),
                range.byte_len,
            )
        }
    }

    /// Wait until all enqueued work on the context's queue has completed: spin on the queue's read
    /// index until it catches up with the write index recorded on entry. Single-dispatch sync; the
    /// signal-based capture/replay sync (`HSA_WAIT_STATE_BLOCKED` + doorbell) is a superset.
    pub fn synchronize(&self) -> Result<(), RocmError> {
        self.ensure_not_poisoned("synchronize")?;
        // Typed construction-time bound (Card 548), matching `wait_completion_bounded`.
        let timeout = self.wait_timeout.as_secs();
        // SAFETY: `queue` is the context's live queue (a `RocmContext` invariant); the index calls only
        // read or atomically bump its indices.
        unsafe {
            let queue = &*self.queue;
            let write_at_submit = (self.hsa.funcs.hsa_queue_add_write_index_relaxed)(queue, 0);
            // Spin on the read index until it reaches our submit count (busy-wait, HSA_WAIT_STATE_ACTIVE
            // semantics).
            let start = std::time::Instant::now();
            let mut spins: u64 = 0;
            loop {
                let read = (self.hsa.funcs.hsa_queue_load_read_index_scacquire)(queue);
                if read >= write_at_submit {
                    break;
                }
                // Wall-clock guard, checked only once the spin has stalled, so the hot path pays nothing. Fails
                // fast on a hung dispatch instead of spinning forever at 99% CPU.
                spins = spins.wrapping_add(1);
                if spins & 0xF_FFFF == 0 && start.elapsed().as_secs() >= timeout {
                    // The queued packets may still run, so no later submission may reuse the queue.
                    self.timeouts.poison();
                    return Err(RocmError::QueueWaitTimeout(timeout, "synchronize"));
                }
                std::hint::spin_loop();
            }
        }
        Ok(())
    }

    /// Pool allocate: `size` bytes from `pool`. Also the primitive for the kernarg buffer (the pool
    /// must have KERNARG_INIT, as `fine_pool` does on Strix Halo).
    pub(crate) fn pool_alloc(
        &self,
        pool: hsa_amd_memory_pool_t,
        size: usize,
    ) -> Result<*mut c_void, RocmError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid out-pointer; the runtime validates `pool`.
        unsafe {
            check((self.hsa.funcs.hsa_amd_memory_pool_allocate)(
                pool, size, 0, &mut ptr,
            ))?;
        }
        Ok(ptr)
    }

    pub(crate) fn resource_owner(&self) -> Arc<dyn ResourceOwner> {
        self.hsa.clone()
    }

    pub(crate) fn own_buffer(
        &self,
        ptr: *mut c_void,
        elem_count: u32,
        byte_capacity: usize,
        storage: BufferStorage,
        role: BufferRole,
        pool: Pool,
    ) -> RocmBuffer {
        RocmBuffer {
            alloc: Arc::new(RocmAlloc {
                ptr: ptr as usize,
                pool,
                owner: self.resource_owner(),
                guard: std::mem::ManuallyDrop::new(self.tag_alloc(role, byte_capacity)),
                poison: self.timeouts.device_poison(),
            }),
            elem_count,
            byte_capacity,
            storage,
        }
    }

    /// Raw access to the loaded function table (for examples + tests that call HSA entry points not yet
    /// wrapped).
    pub fn funcs(&self) -> &Funcs {
        &self.hsa.funcs
    }

    /// The GPU agent this context's queue, code objects and copies target.
    pub fn gpu_agent(&self) -> hsa_agent_t {
        self.gpu_agent
    }

    /// How host<->device transfers reach this context's buffer pool, decided at construction.
    pub fn upload_strategy(&self) -> UploadStrategy {
        self.upload_strategy
    }
}
