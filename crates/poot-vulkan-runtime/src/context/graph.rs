use crate::graph::*;
use crate::*;

/// Descriptor sets a pool is created for. A pool that runs out of sets or descriptors is replaced
/// rather than grown, so a graph of thousands of dispatches needs a few dozen pools, not one per
/// dispatch.
const SETS_PER_POOL: u32 = 256;
/// Descriptors a pool reserves per set; a kernel binding more still fits while the pool has room.
const DESCRIPTORS_PER_SET: u32 = 12;

/// One kernel argument of a dispatch: the buffer, and the element count the kernel is told it holds.
///
/// The count is the plan's own, not the buffer's capacity: an arena slot's buffer is sized to its
/// largest occupant, and a smaller value sharing it must not let a kernel read or write past its own
/// operand (Card 547b; Card 546a F8). A count whose bytes exceed the buffer is refused
/// ([`RuntimeError::ArgExceedsBuffer`]).
#[derive(Clone, Copy)]
pub struct Binding<'a> {
    pub buffer: &'a DeviceBuffer,
    pub elems: u32,
}

impl Context {
    /// Begin recording a new graph (spec 133 P2 slice 2a, FR-008 step 2): allocate a command pool,
    /// the first command buffer and a fence, then `vkBeginCommandBuffer` without `ONE_TIME_SUBMIT` (the
    /// buffers are submitted many times). `SIMULTANEOUS_USE` is not needed because every
    /// [`VulkanGraph::replay`] fence-waits, so two submissions are never in flight at once. Each
    /// following [`Context::record_dispatch`] or [`Context::record_copy`] records one command; call
    /// [`Context::end_graph`] to finish and obtain the replayable [`VulkanGraph`].
    ///
    /// [`GraphTiming::PerDispatch`] adds a timestamp query pool, reset at the start of the first command
    /// buffer, and is refused with [`RuntimeError::TimestampsUnavailable`] on a queue family that
    /// writes none.
    pub fn begin_graph(&self, timing: GraphTiming) -> Result<RecordingGraph, RuntimeError> {
        self.owner.check_live("begin_graph")?;
        if timing == GraphTiming::PerDispatch && !self.timestamps_supported() {
            return Err(RuntimeError::TimestampsUnavailable);
        }
        let resources = create_graph_resources(&self.owner, self.queue_family_index)?;
        let mut core = GraphCore {
            command_pool: resources.command_pool,
            segments: vec![resources.command_buffer],
            fence: resources.fence,
            desc_pools: Vec::new(),
            held: Vec::new(),
            pipelines: Vec::new(),
            meta: MetaArena::default(),
            traps: Vec::new(),
            dispatch_count: 0,
            commands_in_segment: 0,
            timestamps: None,
            owner: Arc::clone(&self.owner),
        };
        if timing == GraphTiming::PerDispatch {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(TIMESTAMP_CAPACITY);
            // SAFETY: `self.device` is valid; `info` outlives the call.
            let pool = unsafe { self.device.create_query_pool(&info, None)? };
            core.timestamps = Some(Timestamps {
                pool,
                written: 0,
                overflowed: false,
                period_ns: self.timestamp_period_ns,
                valid_mask: self.timestamp_valid_mask,
                spans: Vec::new(),
            });
            // SAFETY: the first command buffer is recording (just begun) and outside a render pass;
            // the pool is new and holds `TIMESTAMP_CAPACITY` queries, all reset here before any write.
            unsafe {
                self.device
                    .cmd_reset_query_pool(core.segments[0], pool, 0, TIMESTAMP_CAPACITY);
            }
        }
        Ok(RecordingGraph { core })
    }

    /// Whether this context's queue family writes device timestamps at all.
    pub fn timestamps_supported(&self) -> bool {
        self.timestamp_period_ns > 0.0 && self.timestamp_valid_mask != 0
    }

    /// The cached pipeline for `key`, built from `kernel` on a miss. A hit is returned only for the
    /// same kernel: a key reused for a kernel with different code, argument schema or trap shape is
    /// refused ([`RuntimeError::PipelineKeyConflict`]), so a key can never silently dispatch another
    /// body's pipeline.
    pub fn pipeline(
        &mut self,
        key: &str,
        kernel: &CompiledKernel,
    ) -> Result<Arc<Pipeline>, RuntimeError> {
        self.owner.check_live("pipeline")?;
        if let Some(cached) = self.pipeline_cache.get(key) {
            if !cached.shape.matches(kernel)? {
                return Err(RuntimeError::PipelineKeyConflict {
                    key: key.to_string(),
                });
            }
            return Ok(Arc::clone(cached));
        }
        let built = Arc::new(self.build_pipeline(kernel)?);
        self.pipeline_cache
            .insert(key.to_string(), Arc::clone(&built));
        self.pipeline_builds += 1;
        Ok(built)
    }

    /// Record one dispatch of `pipeline` into `graph`'s open command buffer (spec 133 P2 slice 2a,
    /// FR-007): `args` are the kernel's data buffers, inputs first then the single output last, each
    /// with the element count the kernel is told it holds (see [`Binding`]). The grid is
    /// `ceil(threads / wg)` per axis and must fit the device's per-axis count limits (else
    /// [`RuntimeError::GridCap`]; a grid is never folded); `wg` must match the
    /// module's `LocalSize`.
    ///
    /// Allocates the dispatch's descriptor set (from the graph's pools, kept for the graph's lifetime
    /// since the command buffers are replayed) and its length block, inserts a conservative full memory
    /// barrier if this is not the segment's first command (RAW/WAR/WAW between consecutive commands, as
    /// CUDA graphs assume), and records the pipeline bind, descriptor bind and `vkCmdDispatch`. Never
    /// submits. Every bound buffer and the pipeline are retained by the graph. `key` names the kernel
    /// in a [`KernelFault`].
    ///
    /// The graph, the pipeline and every buffer must belong to this context's device, and `args` must
    /// match the kernel's argument schema (else [`RuntimeError::ForeignObject`] /
    /// [`RuntimeError::KernelArgs`]).
    pub fn record_dispatch(
        &self,
        graph: &mut RecordingGraph,
        key: &str,
        pipeline: &Arc<Pipeline>,
        wg: [u32; 3],
        threads: [u32; 3],
        args: &[Binding<'_>],
    ) -> Result<(), RuntimeError> {
        self.owner.check_live("record_dispatch")?;
        if !Arc::ptr_eq(&graph.core.owner, &self.owner) {
            return Err(RuntimeError::ForeignObject("graph"));
        }
        if !Arc::ptr_eq(&pipeline.owner, &self.owner) {
            return Err(RuntimeError::ForeignObject("pipeline"));
        }
        let groups = self.check_bindings(&pipeline.shape.args, wg, threads, args)?;

        let n = args.len();
        let mut lengths: Vec<u8> = args.iter().flat_map(|a| a.elems.to_le_bytes()).collect();
        lengths.extend_from_slice(&crate::x_extent(groups, wg).to_le_bytes());
        let lengths_block = graph.core.reserve_meta(self, lengths.len())?;
        lengths_block
            .chunk
            .write_bytes_at("record_dispatch", lengths_block.offset, &lengths)?;
        let trap_block = if pipeline.shape.has_trap {
            let block = graph.core.reserve_meta(self, 4)?;
            Some(block)
        } else {
            None
        };
        let desc_set = graph.core.allocate_set(self, pipeline.dsl)?;

        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = args
            .iter()
            .map(|a| {
                vk::DescriptorBufferInfo::default()
                    .buffer(a.buffer.vk_buffer())
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        buffer_infos.push(
            vk::DescriptorBufferInfo::default()
                .buffer(lengths_block.chunk.vk_buffer())
                .offset(lengths_block.offset as vk::DeviceSize)
                .range(lengths_block.len as vk::DeviceSize),
        );
        if let Some(block) = &trap_block {
            buffer_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(block.chunk.vk_buffer())
                    .offset(block.offset as vk::DeviceSize)
                    .range(block.len as vk::DeviceSize),
            );
        }
        debug_assert_eq!(
            buffer_infos.len(),
            n + 1 + usize::from(pipeline.shape.has_trap)
        );
        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        // SAFETY: `writes` (and what they borrow) outlive the call; `self.device` is valid and owns
        // `desc_set` and every bound buffer (`check_bindings` checked their owner); no copies.
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        let core = &mut graph.core;
        let command_buffer = *core.segments.last().expect("a graph has a segment");
        core.barrier_before_command(self, command_buffer);
        let timed = core.timestamp_before(self, command_buffer);
        // SAFETY: `command_buffer` is recording: a `RecordingGraph` exists only between `begin_graph`
        // and the `end_graph` that consumes it, and it belongs to this device. `pipeline.pipeline`,
        // `desc_set` and `pipeline.pipeline_layout` are all valid children of this device and
        // layout-compatible (the descriptor set was allocated from `pipeline.dsl`, the same layout
        // `pipeline.pipeline_layout` was built from).
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            self.device
                .cmd_dispatch(command_buffer, groups[0], groups[1], groups[2]);
        }
        if let Some(start) = timed {
            core.timestamp_after(self, command_buffer, start);
        }
        core.held.extend(args.iter().map(|a| a.buffer.clone()));
        core.pipelines.push(Arc::clone(pipeline));
        if let Some(block) = trap_block {
            core.traps.push(TrapSlot {
                kernel: key.to_string(),
                chunk: block.chunk,
                offset: block.offset,
            });
        }
        core.dispatch_count += 1;
        core.commands_in_segment += 1;
        Ok(())
    }

    /// Record a whole-buffer copy `src -> dst` (`vkCmdCopyBuffer`), ordered after every earlier command
    /// of the graph and before every later one. Both buffers must belong to this device and hold the
    /// same number of bytes.
    pub fn record_copy(
        &self,
        graph: &mut RecordingGraph,
        src: &DeviceBuffer,
        dst: &DeviceBuffer,
    ) -> Result<(), RuntimeError> {
        self.owner.check_live("record_copy")?;
        if !Arc::ptr_eq(&graph.core.owner, &self.owner) {
            return Err(RuntimeError::ForeignObject("graph"));
        }
        if !src.belongs_to(&self.owner) || !dst.belongs_to(&self.owner) {
            return Err(RuntimeError::ForeignObject("buffer"));
        }
        if src.byte_len() != dst.byte_len() {
            return Err(RuntimeError::CopySizeMismatch {
                src: src.byte_len(),
                dst: dst.byte_len(),
            });
        }
        // The same `VkBuffer` as source and destination would overlap, which `vkCmdCopyBuffer` forbids.
        if src.vk_buffer() == dst.vk_buffer() {
            return Err(RuntimeError::CopyOntoItself);
        }
        let core = &mut graph.core;
        let command_buffer = *core.segments.last().expect("a graph has a segment");
        core.barrier_before_command(self, command_buffer);
        let region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(src.byte_len() as vk::DeviceSize);
        // SAFETY: `command_buffer` is recording (see `record_dispatch`); both buffers are distinct
        // children of this device created with TRANSFER_SRC | TRANSFER_DST usage, and the region covers
        // exactly both of them (equal byte lengths were checked above).
        unsafe {
            self.device.cmd_copy_buffer(
                command_buffer,
                src.vk_buffer(),
                dst.vk_buffer(),
                &[region],
            );
        }
        core.held.push(src.clone());
        core.held.push(dst.clone());
        core.commands_in_segment += 1;
        Ok(())
    }

    /// End the graph's open command buffer and start a new one, so a replay submits the commands
    /// before and after as two submissions with a fence wait between them (the display watchdog bounds
    /// one submission's duration, so a caller cuts long graphs by the work it records). A no-op while
    /// the open command buffer is empty.
    pub fn cut_graph(&self, graph: &mut RecordingGraph) -> Result<(), RuntimeError> {
        self.owner.check_live("cut_graph")?;
        if !Arc::ptr_eq(&graph.core.owner, &self.owner) {
            return Err(RuntimeError::ForeignObject("graph"));
        }
        let core = &mut graph.core;
        if core.commands_in_segment == 0 {
            return Ok(());
        }
        let open = *core.segments.last().expect("a graph has a segment");
        // SAFETY: `open` is recording (see `record_dispatch`) and belongs to this device.
        unsafe {
            self.owner
                .dispatch
                .end_command_buffer(&self.owner.device, open)?
        };
        let next = allocate_and_begin(&self.owner, core.command_pool)?;
        core.segments.push(next);
        core.commands_in_segment = 0;
        Ok(())
    }

    /// Finish recording `graph` with `vkEndCommandBuffer` (spec 133 FR-008 step 2) and return the
    /// replayable [`VulkanGraph`]. Consuming the recording graph is what stops any further
    /// [`Context::record_dispatch`] from targeting it. Increments [`Context::graphs_recorded`]. A graph
    /// from another context is refused with [`RuntimeError::ForeignObject`].
    pub fn end_graph(&self, graph: RecordingGraph) -> Result<VulkanGraph, RuntimeError> {
        if !Arc::ptr_eq(&graph.core.owner, &self.owner) {
            return Err(RuntimeError::ForeignObject("graph"));
        }
        self.owner.check_live("end_graph")?;
        let open = *graph.core.segments.last().expect("a graph has a segment");
        // SAFETY: the last command buffer is recording (a `RecordingGraph` is consumed here, so it was
        // opened by `begin_graph` or a cut and never ended), and it belongs to this context's device.
        unsafe {
            self.owner
                .dispatch
                .end_command_buffer(&self.owner.device, open)?;
        }
        self.graphs_recorded.fetch_add(1, Ordering::Relaxed);
        Ok(VulkanGraph::finished(graph))
    }
}

impl GraphCore {
    /// A meta block of `len` bytes, aligned for a descriptor offset, from the arena's last chunk (a
    /// new chunk when it does not fit).
    fn reserve_meta(&mut self, ctx: &Context, len: usize) -> Result<MetaBlock, RuntimeError> {
        let align = ctx.min_storage_offset_alignment.max(4);
        let stride = len.next_multiple_of(align);
        debug_assert!(stride <= META_CHUNK_BYTES);
        let fits = !self.meta.chunks.is_empty() && self.meta.used + stride <= META_CHUNK_BYTES;
        if !fits {
            let chunk = ctx.alloc_storage(
                BufferRole::Meta,
                BufferStorage::dense(ElementKind::RawBytes, LogicalDType::RawBytes),
                META_CHUNK_BYTES,
            )?;
            self.meta.chunks.push(chunk);
            self.meta.used = 0;
        }
        let offset = self.meta.used;
        self.meta.used += stride;
        Ok(MetaBlock {
            chunk: self.meta.chunks.last().expect("a chunk exists").clone(),
            offset,
            len,
        })
    }

    /// A descriptor set of `layout` from the current pool, replacing the pool when it is exhausted.
    fn allocate_set(
        &mut self,
        ctx: &Context,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, RuntimeError> {
        let layouts = [layout];
        if let Some(&pool) = self.desc_pools.last() {
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            // SAFETY: `pool` and `layout` are valid children of this device; `info` outlives the call.
            match unsafe { ctx.device.allocate_descriptor_sets(&info) } {
                Ok(sets) => return Ok(sets[0]),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {}
                Err(e) => return Err(e.into()),
            }
        }
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(SETS_PER_POOL * DESCRIPTORS_PER_SET)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(SETS_PER_POOL)
            .pool_sizes(&sizes);
        // SAFETY: `pool_info` outlives the call; `ctx.device` is valid.
        let pool = unsafe { ctx.device.create_descriptor_pool(&pool_info, None)? };
        // The pool is owned by the graph from here, destroyed with it even if the allocation fails.
        self.desc_pools.push(pool);
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: `pool` was just created and is empty; `layout` is a valid child of this device.
        Ok(unsafe { ctx.device.allocate_descriptor_sets(&info)? }[0])
    }

    /// Order this command after every earlier one of the segment: a conservative full barrier (every
    /// prior shader or transfer write visible to every later shader or transfer access).
    fn barrier_before_command(&self, ctx: &Context, command_buffer: vk::CommandBuffer) {
        if self.commands_in_segment == 0 {
            return;
        }
        let writes = vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE;
        let accesses = writes | vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(writes)
            .dst_access_mask(accesses);
        let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        // SAFETY: `command_buffer` is recording (a `RecordingGraph` exists only between `begin_graph`
        // and the `end_graph` that consumes it) and belongs to this device.
        unsafe {
            ctx.device.cmd_pipeline_barrier(
                command_buffer,
                stages,
                stages,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Write the start timestamp of a dispatch about to be recorded, returning its query index; `None`
    /// for an untimed graph or once the pool is full (which marks the graph's time unknown).
    fn timestamp_before(
        &mut self,
        ctx: &Context,
        command_buffer: vk::CommandBuffer,
    ) -> Option<u32> {
        let ts = self.timestamps.as_mut()?;
        if ts.overflowed || ts.written + 2 > TIMESTAMP_CAPACITY {
            ts.overflowed = true;
            return None;
        }
        let start = ts.written;
        ts.written += 1;
        // SAFETY: `command_buffer` is recording; `start` is a reset query of this graph's own pool.
        unsafe {
            ctx.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                ts.pool,
                start,
            );
        }
        Some(start)
    }

    /// Write the end timestamp of the dispatch whose start was `start`.
    fn timestamp_after(&mut self, ctx: &Context, command_buffer: vk::CommandBuffer, start: u32) {
        let ts = self
            .timestamps
            .as_mut()
            .expect("a start index exists only for a timed graph");
        let end = ts.written;
        ts.written += 1;
        ts.spans.push((start, end));
        // SAFETY: `command_buffer` is recording; `end` is a reset query of this graph's own pool,
        // reserved by the capacity check in `timestamp_before`.
        unsafe {
            ctx.device.cmd_write_timestamp(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                ts.pool,
                end,
            );
        }
    }
}
