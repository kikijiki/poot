use crate::owner::QueueGuard;
use crate::{DeviceBuffer, DeviceOwner, Pipeline, RuntimeError};
use ash::vk;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

pub(crate) struct GraphResources {
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) command_buffer: vk::CommandBuffer,
    pub(crate) fences: [vk::Fence; IN_FLIGHT_SUBMITS],
}

/// How many of a replay's submissions are in flight at once: the device runs one while the next waits
/// behind it, so a segment boundary no longer leaves it idle for the host's wait-reset-submit turnaround,
/// and no more than this much work is ever queued behind the display watchdog (the flush policy bounds
/// each submission, this bounds how many). Each in-flight submission signals its own fence.
pub(crate) const IN_FLIGHT_SUBMITS: usize = 2;

/// Owns an in-progress graph's native resources until construction succeeds.
pub(crate) struct PendingGraph<'a> {
    pub(crate) owner: &'a DeviceOwner,
    pub(crate) command_pool: Option<vk::CommandPool>,
    pub(crate) fences: Vec<vk::Fence>,
}

impl PendingGraph<'_> {
    pub(crate) fn finish(mut self, command_buffer: vk::CommandBuffer) -> GraphResources {
        GraphResources {
            command_pool: self
                .command_pool
                .take()
                .expect("command pool acquired before finish"),
            command_buffer,
            fences: std::mem::take(&mut self.fences)
                .try_into()
                .expect("every fence acquired before finish"),
        }
    }
}

impl Drop for PendingGraph<'_> {
    fn drop(&mut self) {
        // SAFETY: the guard owns every populated handle; destroying the command pool implicitly frees its
        // command buffer.
        unsafe {
            for fence in self.fences.drain(..) {
                self.owner.dispatch.destroy_fence(&self.owner.device, fence);
            }
            if let Some(pool) = self.command_pool.take() {
                self.owner
                    .dispatch
                    .destroy_command_pool(&self.owner.device, pool);
            }
        }
    }
}

pub(crate) fn create_graph_resources(
    owner: &DeviceOwner,
    queue_family_index: u32,
) -> Result<GraphResources, RuntimeError> {
    let mut pending = PendingGraph {
        owner,
        command_pool: None,
        fences: Vec::with_capacity(IN_FLIGHT_SUBMITS),
    };
    let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
    // SAFETY: the owner is valid and outlives the pending resources.
    let command_pool = unsafe {
        owner
            .dispatch
            .create_command_pool(&owner.device, &pool_info)?
    };
    pending.command_pool = Some(command_pool);
    let command_buffer = allocate_command_buffer(owner, command_pool)?;
    let fence_info = vk::FenceCreateInfo::default();
    for _ in 0..IN_FLIGHT_SUBMITS {
        // SAFETY: the owner remains alive and the returned fence transfers to `pending` immediately.
        let fence = unsafe { owner.dispatch.create_fence(&owner.device, &fence_info)? };
        pending.fences.push(fence);
    }
    begin_command_buffer(owner, command_buffer)?;
    Ok(pending.finish(command_buffer))
}

/// A new primary command buffer from `command_pool`, not yet recording.
fn allocate_command_buffer(
    owner: &DeviceOwner,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, RuntimeError> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: `command_pool` is a live pool of this owner's device, owned by the caller.
    Ok(unsafe {
        owner
            .dispatch
            .allocate_command_buffers(&owner.device, &alloc_info)?[0]
    })
}

/// Start recording `command_buffer`, without `ONE_TIME_SUBMIT`: a graph's command buffers are
/// submitted once per replay.
fn begin_command_buffer(
    owner: &DeviceOwner,
    command_buffer: vk::CommandBuffer,
) -> Result<(), RuntimeError> {
    let begin_info = vk::CommandBufferBeginInfo::default();
    // SAFETY: the command buffer was just allocated and is not already recording.
    unsafe {
        owner
            .dispatch
            .begin_command_buffer(&owner.device, command_buffer, &begin_info)?;
    }
    Ok(())
}

/// A new primary command buffer from `command_pool`, already recording.
pub(crate) fn allocate_and_begin(
    owner: &DeviceOwner,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, RuntimeError> {
    let command_buffer = allocate_command_buffer(owner, command_pool)?;
    begin_command_buffer(owner, command_buffer)?;
    Ok(command_buffer)
}

/// How many device timestamps a timed graph reserves: two per dispatch, so a graph measures up to
/// `TIMESTAMP_CAPACITY / 2` dispatches and reports no time at all past that (never a partial sum).
pub(crate) const TIMESTAMP_CAPACITY: u32 = 1 << 13;

/// Whether a graph writes a device timestamp around every dispatch it records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphTiming {
    /// No query pool, no timestamps: replay costs the submit and the fence wait only.
    Off,
    /// Two timestamps per dispatch, read back after each replay's fence wait.
    PerDispatch,
}

/// The device clock readings one graph records around its dispatches (see [`GraphTiming`]).
pub(crate) struct Timestamps {
    pub(crate) pool: vk::QueryPool,
    /// Queries written so far; the next free query index.
    pub(crate) written: u32,
    /// A dispatch did not fit in the pool: this graph's time is unknown, not a partial sum.
    pub(crate) overflowed: bool,
    pub(crate) period_ns: f32,
    /// `timestampValidBits` of the queue family as a mask: bits above it are undefined.
    pub(crate) valid_mask: u64,
    /// `(start, end)` query indices of every timed dispatch, in record order.
    pub(crate) spans: Vec<(u32, u32)>,
}

/// One dispatch's device duration within a replayed graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchTime {
    /// Order among the graph's recorded dispatches (copies are not counted).
    pub index: usize,
    /// The dispatch's own device duration, from its start and end timestamps.
    pub duration: Duration,
}

/// The device time of one replay of a [`GraphTiming::PerDispatch`] graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphDeviceTime {
    /// Sum of each dispatch's own `[start, end)` duration.
    pub sum_of_dispatch_durations: Duration,
    /// First dispatch's start to last dispatch's end on the device clock.
    pub device_span: Duration,
    /// Each timed dispatch's own duration, in record order.
    pub dispatches: Vec<DispatchTime>,
}

/// A kernel assert (`Assert`/`Unreachable`) that fired during a replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelFault {
    /// Name of the kernel that asserted.
    pub kernel: String,
    /// Body-local trap id the kernel wrote.
    pub code: u32,
}

/// What one [`VulkanGraph::replay`] observed besides completing.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Replay {
    /// The first kernel (in record order) whose error word was nonzero after the replay.
    pub fault: Option<KernelFault>,
    /// Present only for a [`GraphTiming::PerDispatch`] graph whose dispatches all fit the query pool.
    pub time: Option<GraphDeviceTime>,
}

/// One trapping dispatch's error word: a 4-byte block of a meta chunk the compiled module writes a
/// nonzero code to, zeroed before every replay and read after it.
pub(crate) struct TrapSlot {
    pub(crate) kernel: String,
    pub(crate) chunk: DeviceBuffer,
    pub(crate) offset: usize,
}

/// The per-dispatch length blocks and error words of one graph, packed into a few large host-visible
/// buffers so a graph of thousands of dispatches costs a handful of device allocations (drivers cap the
/// live allocation count). Blocks are aligned to the device's storage-buffer offset alignment because a
/// descriptor binds each at an offset.
#[derive(Default)]
pub(crate) struct MetaArena {
    pub(crate) chunks: Vec<DeviceBuffer>,
    /// Bytes handed out of the last chunk.
    pub(crate) used: usize,
}

/// The bytes one meta chunk holds.
pub(crate) const META_CHUNK_BYTES: usize = 1 << 20;

/// A block of a meta chunk, as a descriptor binds it.
pub(crate) struct MetaBlock {
    pub(crate) chunk: DeviceBuffer,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

/// The native objects and retained children of one graph, in either state. Its `Drop` releases them;
/// destroying the command pool is valid whether its command buffers are recording or executable, since
/// no submission is ever pending when a graph drops (every [`VulkanGraph::replay`] fence-waits every submission).
pub(crate) struct GraphCore {
    pub(crate) command_pool: vk::CommandPool,
    /// The graph's command buffers in submission order. Each replay submits them one by one, keeping at
    /// most [`IN_FLIGHT_SUBMITS`] pending, so a long graph is never one past-watchdog submission. The last one is the one a
    /// [`RecordingGraph`] is still recording into.
    pub(crate) segments: Vec<vk::CommandBuffer>,
    /// One fence per in-flight submission slot: submission `i` of a replay signals `fences[i % IN_FLIGHT_SUBMITS]`.
    pub(crate) fences: [vk::Fence; IN_FLIGHT_SUBMITS],
    /// Descriptor pools in creation order; the last is the one new sets come from.
    pub(crate) desc_pools: Vec<vk::DescriptorPool>,
    /// Buffers the recorded command buffers reference, kept alive for the graph's lifetime: every
    /// buffer a recorded dispatch or copy binds.
    pub(crate) held: Vec<DeviceBuffer>,
    /// Pipelines referenced by recorded binds. The context cache may disappear first.
    pub(crate) pipelines: Vec<Arc<Pipeline>>,
    pub(crate) meta: MetaArena,
    pub(crate) traps: Vec<TrapSlot>,
    /// Number of dispatches recorded; constant across replays.
    pub(crate) dispatch_count: usize,
    /// Commands recorded into the open segment: a barrier precedes every command but the graph's first
    /// (a segment's first command is ordered after the earlier segments by that barrier, since their
    /// submissions may still be running).
    pub(crate) commands_in_segment: usize,
    pub(crate) timestamps: Option<Timestamps>,
    /// Last so graph children, held buffers, and retained pipelines drop before the device owner.
    pub(crate) owner: Arc<DeviceOwner>,
}

/// A graph whose command buffers are still recording (spec 133 P2 slice 2a, card 139): returned by
/// [`Context::begin_graph`], filled by [`Context::record_dispatch`] and [`Context::record_copy`], and
/// consumed by [`Context::end_graph`], which returns the replayable [`VulkanGraph`]. It has no `replay`:
/// submitting a command buffer in the recording state is invalid Vulkan usage, so the state is a type,
/// not a check.
pub struct RecordingGraph {
    pub(crate) core: GraphCore,
}

/// A finished record-once/replay-many command-buffer list (spec 133 P2 slice 2a, card 139), the Vulkan
/// analogue of CUDA graphs / ROCm AQL replay, using Vulkan's native command-buffer resubmit. Only
/// [`Context::end_graph`] builds one, from a [`RecordingGraph`], so its command buffers are always in
/// the executable state; replayed by [`VulkanGraph::replay`].
///
/// The graph shares the loader/instance/device owner, every buffer its descriptor sets reference, and
/// every pipeline baked into its command buffers, so it can be replayed and dropped after the original
/// [`Context`] is gone.
pub struct VulkanGraph {
    core: GraphCore,
    /// Held for a whole replay: the graph's fences, trap words and timestamp pool serve one replay
    /// at a time, however many threads share the graph.
    replaying: Mutex<()>,
}

impl VulkanGraph {
    /// The finished form of a graph whose last command buffer `vkEndCommandBuffer` just closed.
    pub(crate) fn finished(recording: RecordingGraph) -> Self {
        VulkanGraph {
            core: recording.core,
            replaying: Mutex::new(()),
        }
    }

    /// Number of dispatches in the graph; constant across replays.
    pub fn dispatch_count(&self) -> usize {
        self.core.dispatch_count
    }

    /// Number of command buffers a replay submits, [`IN_FLIGHT_SUBMITS`] of them in flight at once.
    #[cfg(any(test, feature = "test-support"))]
    pub fn segment_count(&self) -> usize {
        self.core.segments.len()
    }

    /// Re-submit the recorded command buffers in order, keeping at most [`IN_FLIGHT_SUBMITS`] pending
    /// (spec 133 FR-009): before a submission would exceed the window the host waits for the oldest, so
    /// the device runs one segment while the next is already queued behind it. Waiting after every
    /// segment left the device idle for the host's wake-reset-submit turnaround once per segment
    /// (POOT-1449). The last submissions are waited out before the call returns.
    /// Write new per-replay input data into the bound persistent buffers (via
    /// [`DeviceBuffer::write_with`]) before calling. Nothing is re-recorded.
    ///
    /// Every trapping kernel's error word is zeroed before the first submit and read after the last
    /// wait: a kernel that asserted comes back as [`Replay::fault`], a program result and not a device
    /// failure. A timed graph reads its timestamps back in the same call, after the waits.
    ///
    /// A replay whose fence wait fails (or whose submit is lost) returns
    /// [`RuntimeError::SubmissionLost`] and poisons the device: a command buffer may still be pending,
    /// so every later replay is refused ([`RuntimeError::Poisoned`]) before it touches a fence, and
    /// dropping the graph leaks its objects. A submit refused for memory leaves the device usable, so the
    /// submissions already pending are waited out before the error returns. Replays of one graph
    /// serialize on its lock (the fences are reused by every replay).
    pub fn replay(&self) -> Result<Replay, RuntimeError> {
        let core = &self.core;
        core.owner.check_live("replay")?;
        let _replaying = self
            .replaying
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // The whole replay holds the device: no host copy through a buffer's mapping can overlap the GPU
        // access of any submission below, and none can run until the last fence wait returns.
        let queue = core.owner.lock_queue();
        for trap in &core.traps {
            trap.chunk
                .write_with_at(&queue, "replay", trap.offset, 4, &mut |_, out| out.fill(0))?;
        }
        // A refused submit queued nothing, but the earlier ones are still running; the caller may destroy
        // the graph on that error, so they are retired either way.
        let mut submitted = 0;
        let outcome = core.submit_segments(&queue, &mut submitted);
        let retired = core.wait_pending(submitted);
        outcome?;
        retired?;
        Ok(Replay {
            fault: core.read_fault(&queue)?,
            time: core.read_time()?,
        })
    }
}

impl GraphCore {
    /// Submit every segment in order, retiring the oldest submission first whenever the window is full.
    /// `submitted` counts the segments queued so far, also when a later one fails.
    ///
    /// Submission `i` signals `fences[i % IN_FLIGHT_SUBMITS]`, so the fence a submission reuses belongs to
    /// the one retired just before it.
    fn submit_segments(
        &self,
        queue: &QueueGuard<'_>,
        submitted: &mut usize,
    ) -> Result<(), RuntimeError> {
        for (index, &command_buffer) in self.segments.iter().enumerate() {
            let fence = self.fences[index % IN_FLIGHT_SUBMITS];
            if index >= IN_FLIGHT_SUBMITS {
                // SAFETY: `fence` signals submission `index - IN_FLIGHT_SUBMITS` of this replay, queued
                // below; the caller holds the device's queue guard.
                unsafe { self.owner.wait_submission(fence)? };
            }
            // SAFETY: `fence` is this graph's own, and no submission signalling it is pending: the one
            // that used it last (the previous replay's, or `index - IN_FLIGHT_SUBMITS`'s) was waited just
            // above or before the previous replay returned, or poisoned the device, which `check_live`
            // refused in `replay` and `wait_submission` reports. Resetting an already-unsignaled fence
            // (the first replay) is a valid no-op. `owner` retains the complete native parent chain.
            unsafe {
                self.owner
                    .dispatch
                    .reset_fences(&self.owner.device, &[fence])?;
            }
            // SAFETY: `command_buffer` is executable, since a `VulkanGraph` is only built by
            // `end_graph` after `vkEndCommandBuffer` closed the last segment (and every earlier one was
            // closed by the cut that opened its successor), and it is not pending: each segment is
            // submitted once per replay and every submission of an earlier replay was waited out;
            // `fence` was just reset. Every buffer and pipeline the command buffers reference is
            // retained by `self`, and leaked with it if a submission is lost.
            unsafe { self.owner.submit(queue, command_buffer, fence)? };
            *submitted += 1;
        }
        Ok(())
    }

    /// Wait for the submissions still pending after `submitted` of this replay's segments were queued: the
    /// last [`IN_FLIGHT_SUBMITS`] of them. Nothing is waited for once the device is poisoned: a lost
    /// submission or a failed wait already reported, and waiting again would only outlast the bound once
    /// more.
    fn wait_pending(&self, submitted: usize) -> Result<(), RuntimeError> {
        if self.owner.is_poisoned() {
            return Ok(());
        }
        for index in submitted.saturating_sub(IN_FLIGHT_SUBMITS)..submitted {
            // SAFETY: submission `index` of this replay was queued with `fences[index %
            // IN_FLIGHT_SUBMITS]` and has not been waited yet (only earlier ones were); the caller holds
            // the device's queue guard.
            unsafe {
                self.owner
                    .wait_submission(self.fences[index % IN_FLIGHT_SUBMITS])?
            };
        }
        Ok(())
    }

    /// The first trapping kernel whose error word is nonzero.
    fn read_fault(&self, queue: &QueueGuard<'_>) -> Result<Option<KernelFault>, RuntimeError> {
        for trap in &self.traps {
            let mut word = [0u8; 4];
            trap.chunk
                .read_bytes_at(queue, "replay", trap.offset, &mut word)?;
            let code = u32::from_le_bytes(word);
            if code != 0 {
                return Ok(Some(KernelFault {
                    kernel: trap.kernel.clone(),
                    code,
                }));
            }
        }
        Ok(None)
    }

    /// The timestamps of the replay that just completed, or `None` for an untimed graph, an empty one,
    /// or one with more dispatches than the query pool holds.
    fn read_time(&self) -> Result<Option<GraphDeviceTime>, RuntimeError> {
        let Some(ts) = &self.timestamps else {
            return Ok(None);
        };
        if ts.overflowed || ts.spans.is_empty() {
            return Ok(None);
        }
        let mut ticks = vec![0u64; ts.written as usize];
        // SAFETY: `ts.pool` is this graph's own pool; its first `written` queries were each written by
        // the command buffers whose fence waits completed above, so every result is available;
        // `WAIT` makes that explicit.
        unsafe {
            self.owner.device.get_query_pool_results(
                ts.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        let nanos = |start: u32, end: u32| {
            let delta = ticks[end as usize].wrapping_sub(ticks[start as usize]) & ts.valid_mask;
            Duration::from_nanos((delta as f64 * f64::from(ts.period_ns)) as u64)
        };
        let dispatches: Vec<DispatchTime> = ts
            .spans
            .iter()
            .enumerate()
            .map(|(index, &(start, end))| DispatchTime {
                index,
                duration: nanos(start, end),
            })
            .collect();
        let first = ts.spans[0].0;
        let last = ts.spans[ts.spans.len() - 1].1;
        Ok(Some(GraphDeviceTime {
            sum_of_dispatch_durations: dispatches.iter().map(|d| d.duration).sum(),
            device_span: nanos(first, last),
            dispatches,
        }))
    }
}

impl Drop for GraphCore {
    fn drop(&mut self) {
        if self.owner.is_poisoned() {
            // A lost replay may leave a command buffer pending: leak the pools, fence, query pool and
            // command buffers. `held` and `pipelines` still drop, and their own `Drop`s leak for the
            // same reason.
            return;
        }
        // SAFETY: no GPU work referencing this graph's objects can be in flight (every `replay()`
        // fence-waits every submission, and a failed wait poisoned the device, which returned above), so every object
        // below is safe to destroy, and command buffers still in the recording state (a dropped
        // `RecordingGraph`) may be freed with their pool. `owner` keeps the device alive through these
        // calls. `desc_pools` and the query pool go before `command_pool` (whose destruction frees the
        // command buffers) to keep child-before-parent order. `held` drops afterwards in normal
        // field-drop order, each buffer via its own `Drop`; the order against the destroys below does
        // not matter since nothing is pending.
        unsafe {
            for &fence in &self.fences {
                self.owner.dispatch.destroy_fence(&self.owner.device, fence);
            }
            for &pool in &self.desc_pools {
                self.owner
                    .dispatch
                    .destroy_descriptor_pool(&self.owner.device, pool);
            }
            if let Some(ts) = &self.timestamps {
                self.owner.device.destroy_query_pool(ts.pool, None);
            }
            self.owner
                .dispatch
                .destroy_command_pool(&self.owner.device, self.command_pool);
        }
    }
}
