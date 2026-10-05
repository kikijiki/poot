use crate::*;

impl Context {
    /// Read a device buffer back to host f32 (staging copy).
    ///
    /// Card 531c: this is the resident/cached path's step boundary in practice - the
    /// final logits/argmax-index readback of a decode step - so it also flushes and checks every
    /// [`crate::pipeline::PendingFault`] staged since the last such flush, in the same submission as its
    /// own copy (no extra `queue.submit` or `poll`).
    pub fn download_f32(&self, b: &DeviceBuffer) -> Result<Vec<f32>, RuntimeError> {
        self.readbacks.set(self.readbacks.get() + 1);
        self.readback_staging_allocs
            .set(self.readback_staging_allocs.get() + 1);
        let size = (b.elem_count as usize * 4) as wgpu::BufferAddress;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // Card 547a: direct result/validation readback staging, held for `staging`'s lifetime.
        let _staging_guard = self.tag_alloc(BufferRole::Staging, size as usize);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&b.buf, 0, &staging, 0, size);
        let staged_faults = self.stage_pending_faults(&mut enc);
        // NOTE (G6): this is its own `queue.submit`, separate from a preceding `submit_encoded`, and
        // not counted by `self.submits` (only `submit_encoded`/`dispatch`/`dispatch_dev` bump it). A
        // cached decode step is therefore at least 2 real wgpu submits (compute batch + this readback
        // copy), not the 1 the "SC-002" doc comments describe; the on-device-argmax variant adds a third
        // (the argmax kernel's `dispatch_dev`) plus this one again for the index.
        self.native_submits.set(self.native_submits.get() + 1);
        self.record_submit(poot_runtime_common::CallPurpose::Readback);
        self.queue.submit(Some(enc.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        for (_, fs, _) in &staged_faults {
            fs.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        self.waits.set(self.waits.get() + 1);
        self.record_wait(poot_runtime_common::CallPurpose::Readback);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RuntimeError::Readback(e.to_string()))?;
        let data = staging.slice(..).get_mapped_range();
        self.record_transfer(
            poot_runtime_common::CallPurpose::Readback,
            poot_runtime_common::TransferDirection::DeviceToHost,
            size as u64,
        );
        let result = bytemuck::cast_slice(&data).to_vec();
        self.check_staged_faults(&staged_faults)?;
        Ok(result)
    }

    /// Read the leading lanes of several device buffers back as one concatenated host word vector.
    ///
    /// All segments are copied into one staging buffer by one command encoder, mapped once, and counted
    /// as one readback, so a validation packet's bounded witness lanes cost a single readback however
    /// many witness buffers the graph declares. An empty request reads nothing.
    ///
    /// `words` and the source length guard are 4-byte lanes, so a segment must name a buffer whose
    /// `elem_count` counts 4-byte elements (f32/i32/u32 uploads and allocations, not
    /// [`Self::upload_f16`], whose `elem_count` counts 2-byte elements). A validation packet is F32 by
    /// construction.
    ///
    /// Card 531c: also flushes and checks every pending fault staged since the last
    /// flush, in the same submission as its own copies - except on the `size == 0` fast path below,
    /// which returns before any encoder exists; a step whose only readback is an empty segment list
    /// (no witness lanes requested) is not this crate's step-boundary sync and must reach one of the
    /// other `download_f32`/`read_bytes`/`flush_faults_and_wait` calls to have its faults checked.
    ///
    /// `pub(crate)`/test-only: no production caller survives Card 546b's `GpuExecutor` deletion;
    /// kept for the runtime-level memory-accounting test (`src/tests.rs`).
    #[cfg(test)]
    pub(crate) fn download_segments_u32(
        &self,
        segments: &[BufferSegment<'_>],
    ) -> Result<Vec<u32>, RuntimeError> {
        let (offsets, size) = segment_byte_layout(
            segments
                .iter()
                .map(|segment| (segment.words, segment.buffer.elem_count as usize)),
        )?;
        if size == 0 {
            return Ok(Vec::new());
        }
        self.readbacks.set(self.readbacks.get() + 1);
        self.readback_staging_allocs
            .set(self.readback_staging_allocs.get() + 1);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_segments"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // Card 547a: direct result/validation readback staging, held for `staging`'s lifetime.
        let _staging_guard = self.tag_alloc(BufferRole::Staging, size as usize);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (segment, &offset) in segments.iter().zip(&offsets) {
            if segment.words != 0 {
                enc.copy_buffer_to_buffer(
                    &segment.buffer.buf,
                    0,
                    &staging,
                    offset,
                    segment.words as u64 * 4,
                );
            }
        }
        let staged_faults = self.stage_pending_faults(&mut enc);
        self.native_submits.set(self.native_submits.get() + 1);
        self.record_submit(poot_runtime_common::CallPurpose::Readback);
        self.queue.submit(Some(enc.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        for (_, fs, _) in &staged_faults {
            fs.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        self.waits.set(self.waits.get() + 1);
        self.record_wait(poot_runtime_common::CallPurpose::Readback);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RuntimeError::Readback(e.to_string()))?;
        let data = staging.slice(..).get_mapped_range();
        self.record_transfer(
            poot_runtime_common::CallPurpose::Readback,
            poot_runtime_common::TransferDirection::DeviceToHost,
            size,
        );
        let result = bytemuck::cast_slice(&data).to_vec();
        self.check_staged_faults(&staged_faults)?;
        Ok(result)
    }

    /// Read the first `out.len()` bytes of a device buffer (Card 546a: the storage-neutral read the
    /// executor contract's binder and `StepOutputs::read` need). Flushes and checks every staged
    /// kernel-assert fault in the same submission as its own copy, like [`Self::download_f32`].
    ///
    /// Card 547a: the device-side copy is padded up to `COPY_BUFFER_ALIGNMENT` (4 bytes) when
    /// `out.len()` is not already a multiple of 4 (a BF16/F16 buffer with an odd element count),
    /// mirroring [`Self::write_bytes`]'s own padding and `alloc_storage`'s padded allocation (which
    /// guarantees the source buffer holds at least that many bytes); only `out.len()` bytes are copied
    /// back into `out`.
    ///
    /// Card 547a: `out` longer than `b`'s byte capacity is refused with the typed
    /// [`RuntimeError::RangeOutOfBounds`]/[`RuntimeError::RangeOverflow`] (mirroring
    /// [`Self::write_bytes`]'s own `checked_write_range` call), and the device-side copy is capped at
    /// `b`'s own padded byte capacity - never past it, even though both are already multiples of 4 by
    /// construction (`alloc_storage` pads the same way), so a future allocator that pads differently
    /// cannot reopen the overrun this closes.
    pub fn read_bytes(&self, b: &DeviceBuffer, out: &mut [u8]) -> Result<(), RuntimeError> {
        checked_write_range("read_bytes", b, 0, out.len(), 1)?;
        self.readbacks.set(self.readbacks.get() + 1);
        self.readback_staging_allocs
            .set(self.readback_staging_allocs.get() + 1);
        let byte_capacity = b.elem_count() as usize * b.storage().element().byte_width();
        let padded_capacity = byte_capacity.div_ceil(4) * 4;
        let size = (out.len().div_ceil(4) * 4).min(padded_capacity) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_bytes"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // Card 547a: direct result/validation readback staging, held for `staging`'s lifetime.
        let _staging_guard = self.tag_alloc(BufferRole::Staging, size as usize);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&b.buf, 0, &staging, 0, size);
        let staged_faults = self.stage_pending_faults(&mut enc);
        self.native_submits.set(self.native_submits.get() + 1);
        self.record_submit(poot_runtime_common::CallPurpose::Readback);
        self.queue.submit(Some(enc.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        for (_, fs, _) in &staged_faults {
            fs.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        self.waits.set(self.waits.get() + 1);
        self.record_wait(poot_runtime_common::CallPurpose::Readback);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RuntimeError::Readback(e.to_string()))?;
        {
            let data = staging.slice(..).get_mapped_range();
            out.copy_from_slice(&data[..out.len()]);
        }
        self.record_transfer(
            poot_runtime_common::CallPurpose::Readback,
            poot_runtime_common::TransferDirection::DeviceToHost,
            out.len() as u64,
        );
        self.check_staged_faults(&staged_faults)?;
        Ok(())
    }

    /// Wait for submitted work and raise any staged kernel-assert fault (Card 531c, Card 546a
    /// S46-11): the `Device::synchronize` seam that reads no data back, unlike every other fault
    /// check here which rides a real readback's submission.
    pub fn flush_faults_and_wait(&self) -> Result<(), RuntimeError> {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let staged_faults = self.stage_pending_faults(&mut enc);
        self.native_submits.set(self.native_submits.get() + 1);
        self.record_submit(poot_runtime_common::CallPurpose::Validation);
        self.queue.submit(Some(enc.finish()));
        for (_, fs, _) in &staged_faults {
            fs.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        self.waits.set(self.waits.get() + 1);
        self.record_wait(poot_runtime_common::CallPurpose::Validation);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RuntimeError::Readback(e.to_string()))?;
        self.check_staged_faults(&staged_faults)
    }

    /// Copy every 4-byte lane of `src` into `dst` on the device (Card 375f).
    ///
    /// A validated state session seeds its own storage from a caller's state buffer this way instead of
    /// adopting the buffer, so the caller keeps no handle into session storage. One submit, no readback,
    /// no allocation. Both buffers must hold the same number of lanes.
    pub fn copy_words_into(
        &self,
        dst: &DeviceBuffer,
        src: &DeviceBuffer,
    ) -> Result<(), RuntimeError> {
        if dst.elem_count != src.elem_count {
            return Err(RuntimeError::CopyLength {
                dst: dst.elem_count,
                src: src.elem_count,
            });
        }
        // Card 546a: the element width comes from the buffer's own storage (card 527), not a
        // hardcoded 4 bytes - this copy now also serves the executor contract's two-phase state
        // commit, whose state buffers are not always F32/I32-wide.
        let bytes = src.elem_count as u64 * src.storage.element().byte_width() as u64;
        if bytes == 0 {
            return Ok(());
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&src.buf, 0, &dst.buf, 0, bytes);
        // Not a compute submit, so it stays out of `submits` (the launch-tax signal), like the readback and
        // upload submits around it.
        self.native_submits.set(self.native_submits.get() + 1);
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }
}
