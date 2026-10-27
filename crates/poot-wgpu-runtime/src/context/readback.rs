use crate::buffer::checked_write_range;
use crate::{Context, DeviceBuffer, RuntimeError};
use poot_runtime_common::BufferRole;
use std::sync::{Arc, Mutex};

/// The outcomes of the `map_async` calls one readback registered. wgpu reports a failed map to the
/// callback only, and reading a range that failed to map panics, so a readback keeps each result and
/// checks them all after its wait, before it touches a mapped range.
#[derive(Default)]
pub(crate) struct MapSet {
    results: Vec<MapResult>,
}

/// What a map callback delivered; `None` until it has run.
type MapResult = Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>;

impl MapSet {
    /// Map `slice` for reading and remember how that went.
    pub(crate) fn read(&mut self, slice: wgpu::BufferSlice<'_>) {
        let result = Arc::new(Mutex::new(None));
        let callback = Arc::clone(&result);
        slice.map_async(wgpu::MapMode::Read, move |mapped| {
            *callback
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(mapped);
        });
        self.results.push(result);
    }

    /// The first failed map, as a typed error; a map whose callback never ran is a failure too.
    pub(crate) fn check(&self) -> Result<(), RuntimeError> {
        for result in &self.results {
            match result
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
            {
                Some(Ok(())) => {}
                Some(Err(error)) => return Err(RuntimeError::MapFailed(error)),
                None => return Err(RuntimeError::MapNotCompleted),
            }
        }
        Ok(())
    }
}

impl Context {
    /// Read a device buffer back to host f32 (staging copy).
    ///
    /// Card 531c: this is the resident/cached path's step boundary in practice - the
    /// final logits/argmax-index readback of a decode step - so it also flushes and checks every
    /// [`crate::pipeline::PendingFault`] staged since the last such flush, in the same submission as its
    /// own copy (no extra `queue.submit` or `poll`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn download_f32(&self, b: &DeviceBuffer) -> Result<Vec<f32>, RuntimeError> {
        self.readbacks.set(self.readbacks.get() + 1);
        self.readback_staging_allocs
            .set(self.readback_staging_allocs.get() + 1);
        let size = (b.elem_count * 4) as wgpu::BufferAddress;
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
        let mut maps = MapSet::default();
        maps.read(staging.slice(..));
        for (_, fs, _) in &staged_faults {
            maps.read(fs.slice(..));
        }
        self.waits.set(self.waits.get() + 1);
        self.record_wait(poot_runtime_common::CallPurpose::Readback);
        self.wait_idle()?;
        maps.check()?;
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

    /// Read the first `out.len()` bytes of a device buffer (Card 546a: the storage-neutral read the
    /// executor contract's binder and `StepOutputs::read` need). Flushes and checks every staged
    /// kernel-assert fault in the same submission as its own copy, like [`Self::download_f32`].
    ///
    /// Card 547a: the device-side copy is padded up to `COPY_BUFFER_ALIGNMENT` (4 bytes) when
    /// `out.len()` is not already a multiple of 4 (a BF16/F16 buffer with an odd element count),
    /// mirroring [`Self::write_with`]'s own padding and `alloc_storage`'s padded allocation (which
    /// guarantees the source buffer holds at least that many bytes); only `out.len()` bytes are copied
    /// back into `out`.
    ///
    /// Card 547a: `out` longer than `b`'s byte capacity is refused with the typed
    /// [`RuntimeError::RangeOutOfBounds`]/[`RuntimeError::RangeOverflow`] (mirroring
    /// [`Self::write_with`]'s own `checked_write_range` call), and the device-side copy is capped at
    /// `b`'s own padded byte capacity - never past it, even though both are already multiples of 4 by
    /// construction (`alloc_storage` pads the same way), so a future allocator that pads differently
    /// cannot reopen the overrun this closes.
    pub fn read_bytes(&self, b: &DeviceBuffer, out: &mut [u8]) -> Result<(), RuntimeError> {
        checked_write_range("read_bytes", b, 0, out.len(), 1)?;
        self.readbacks.set(self.readbacks.get() + 1);
        self.readback_staging_allocs
            .set(self.readback_staging_allocs.get() + 1);
        let byte_capacity = b.elem_count() * b.storage().element().byte_width();
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
        let mut maps = MapSet::default();
        maps.read(staging.slice(..));
        for (_, fs, _) in &staged_faults {
            maps.read(fs.slice(..));
        }
        self.waits.set(self.waits.get() + 1);
        self.record_wait(poot_runtime_common::CallPurpose::Readback);
        self.wait_idle()?;
        maps.check()?;
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
        let mut maps = MapSet::default();
        for (_, fs, _) in &staged_faults {
            maps.read(fs.slice(..));
        }
        self.waits.set(self.waits.get() + 1);
        self.record_wait(poot_runtime_common::CallPurpose::Validation);
        self.wait_idle()?;
        maps.check()?;
        self.check_staged_faults(&staged_faults)
    }

    /// The number of bytes a whole-buffer copy of `src` into `dst` moves: the allocation of either, which
    /// is its elements rounded up to whole 4-byte words (`alloc_storage`), so a buffer of an odd number of
    /// bytes (3 bf16 values) copies whole. Refused with [`RuntimeError::CopyLength`] when the two hold
    /// different byte lengths.
    pub fn copy_bytes(&self, dst: &DeviceBuffer, src: &DeviceBuffer) -> Result<u64, RuntimeError> {
        let bytes = |buffer: &DeviceBuffer| {
            (buffer.elem_count as u64 * buffer.storage.element().byte_width() as u64)
                .next_multiple_of(4)
        };
        let (dst_bytes, src_bytes) = (bytes(dst), bytes(src));
        if dst_bytes != src_bytes {
            return Err(RuntimeError::CopyLength {
                dst: dst_bytes,
                src: src_bytes,
            });
        }
        Ok(src_bytes)
    }

    /// Copy all of `src` into `dst` on the device (Card 375f), whatever the buffers' element kind or length
    /// ([`Self::copy_bytes`]).
    ///
    /// A validated state session seeds its own storage from a caller's state buffer this way instead of
    /// adopting the buffer, so the caller keeps no handle into session storage. One submit, no readback,
    /// no allocation. Both buffers must hold the same number of bytes.
    pub fn copy_words_into(
        &self,
        dst: &DeviceBuffer,
        src: &DeviceBuffer,
    ) -> Result<(), RuntimeError> {
        let bytes = self.copy_bytes(dst, src)?;
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

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;
    use crate::buffer::test_only::{BufferSegment, segment_byte_layout};

    impl Context {
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
        pub(crate) fn download_segments_u32(
            &self,
            segments: &[BufferSegment<'_>],
        ) -> Result<Vec<u32>, RuntimeError> {
            let (offsets, size) = segment_byte_layout(
                segments
                    .iter()
                    .map(|segment| (segment.words, segment.buffer.elem_count)),
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
            let mut maps = MapSet::default();
            maps.read(staging.slice(..));
            for (_, fs, _) in &staged_faults {
                maps.read(fs.slice(..));
            }
            self.waits.set(self.waits.get() + 1);
            self.record_wait(poot_runtime_common::CallPurpose::Readback);
            self.wait_idle()?;
            maps.check()?;
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
    }
}
