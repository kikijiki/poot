use crate::*;

impl Context {
    // Device-resident path: keep intermediates on the GPU across dispatches (no per-op host round-trip).
    // Upload inputs once, dispatch over persistent buffers, download only the final output. This is the
    // decode speedup; the per-op `dispatch` above stays for simple cases.

    /// Upload f32 data to a persistent device buffer.
    pub fn upload_f32(&self, data: &[f32]) -> DeviceBuffer {
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dev"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        self.record_upload_transfer(std::mem::size_of_val(data));
        DeviceBuffer {
            buf,
            elem_count: data.len() as u32,
            id: next_buffer_id(),
            storage: BufferStorage::f32(),
            _guard: self.tag_alloc(
                poot_runtime_common::BufferRole::Weight,
                std::mem::size_of_val(data),
            ),
        }
    }

    /// Upload exact i32 bytes as a device buffer, for a kernel param declared `Slice<i32>` (a
    /// `PackedDequant`/`PackedContraction` carrier or an index buffer). The storage buffer is
    /// dtype-agnostic bytes; the SPIR-V declares the i32 interpretation.
    pub fn upload_i32(&self, data: &[i32]) -> DeviceBuffer {
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dev_i32"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        self.record_upload_transfer(std::mem::size_of_val(data));
        DeviceBuffer {
            buf,
            elem_count: data.len() as u32,
            id: next_buffer_id(),
            storage: BufferStorage::i32(),
            _guard: self.tag_alloc(
                poot_runtime_common::BufferRole::Weight,
                std::mem::size_of_val(data),
            ),
        }
    }

    /// Upload raw IEEE-754 half-precision bits as a device buffer, for a kernel param declared
    /// `Slice<f16>` (card 154's SPIR-V cooperative-matrix matmul operands). Each `u16` is a narrowed
    /// f16's bit pattern (e.g. from `poot_quant::scalar::f32_to_f16`), little-endian, 2 bytes per element;
    /// `upload_f32` would upload the same element count as 4-byte words, which the coopmat kernel's
    /// byte-addressed `Slice<f16>` binding would index at the wrong stride.
    pub fn upload_f16(&self, data: &[u16]) -> DeviceBuffer {
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dev_f16"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        self.record_upload_transfer(std::mem::size_of_val(data));
        DeviceBuffer {
            buf,
            elem_count: data.len() as u32,
            id: next_buffer_id(),
            storage: BufferStorage::f16(),
            _guard: self.tag_alloc(
                poot_runtime_common::BufferRole::Weight,
                std::mem::size_of_val(data),
            ),
        }
    }

    /// Upload u32 data to a device buffer, for a kernel param the body declares as `Slice<u32>`, e.g. a
    /// shape-generic kernel's runtime dimensions (`dims: &[u32]`). Same storage as the i32/f32 forms.
    pub fn upload_u32(&self, data: &[u32]) -> DeviceBuffer {
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dev_u32"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        self.record_upload_transfer(std::mem::size_of_val(data));
        DeviceBuffer {
            buf,
            elem_count: data.len() as u32,
            id: next_buffer_id(),
            // No fixed tensor dtype: this generic word buffer serves dims metadata, seeds, scratch
            // and packed rows alike, each named by the caller, not this allocation (card 527).
            storage: BufferStorage::dense(ElementKind::I32, LogicalDType::RawBytes),
            _guard: self.tag_alloc(
                poot_runtime_common::BufferRole::Meta,
                std::mem::size_of_val(data),
            ),
        }
    }

    /// Upload u32 data to a persistent device buffer that can later be rewritten in place.
    /// E4M3FN slots use this for their row-padded packed-word storage. `pub(crate)`: no production
    /// caller survives Card 546b's `GpuExecutor` deletion; kept for the runtime-level
    /// `submit_cached` fault/memory tests (`src/tests.rs`), which need a writable buffer independent
    /// of any executor.
    #[cfg(test)]
    pub(crate) fn upload_u32_writable(&self, data: &[u32]) -> DeviceBuffer {
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dev_slot_u32"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        self.record_upload_transfer(std::mem::size_of_val(data));
        DeviceBuffer {
            buf,
            elem_count: data.len() as u32,
            id: next_buffer_id(),
            // Card 527: E4M3FN packed rows are this function's documented caller; other packed or
            // metadata payloads may reuse it too, so the logical dtype stays opaque here as well.
            storage: BufferStorage::dense(ElementKind::I32, LogicalDType::RawBytes),
            _guard: self.tag_alloc(
                poot_runtime_common::BufferRole::Weight,
                std::mem::size_of_val(data),
            ),
        }
    }

    /// A fresh uninitialized persistent device buffer of `elems` f32, for a dispatch's output, which
    /// every poot-gpu caller fully overwrites before anything reads it (one compute kernel writes every
    /// output element; `DynamicUpdateSlice`'s kernel materializes base-or-update per element). Card 138:
    /// going through `upload_f32(&vec![0.0; elems])` allocated and zero-filled a host `Vec` and copied it
    /// via `create_buffer_init`, an H2D transfer of a value nothing reads; with ~1 output
    /// buffer per dispatch and hundreds of dispatches/token on the un-fused decode path it dominated the
    /// wgpu quant path's per-token H2D bytes (qwen2.5-1.5B Q4_K_M dropped from ~9 MB/token to a few
    /// KB/token). Do not use this for a buffer whose zero content matters (e.g. a KV-cache seed); use
    /// [`Context::upload_f32`] with an explicit zero vector.
    pub fn alloc_f32(&self, elems: usize) -> DeviceBuffer {
        self.alloc_f32_labeled(elems, "dev_out")
    }

    /// Same as [`Context::alloc_f32`], with a caller-supplied wgpu debug label instead of the generic
    /// `"dev_out"`. Card 158 diagnostic: `exec_resident` labels each dispatch's output buffer with its
    /// plan `key` (the op+shape cache key), so a deferred wgpu validation error naming this buffer's
    /// `ResourceErrorIdent` (`"{type} with '{label}' label"`) identifies which op produced it.
    pub fn alloc_f32_labeled(&self, elems: usize, label: &str) -> DeviceBuffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (elems * 4) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.buffer_allocs.set(self.buffer_allocs.get() + 1);
        DeviceBuffer {
            buf,
            elem_count: elems as u32,
            id: next_buffer_id(),
            storage: BufferStorage::f32(),
            _guard: self.tag_alloc(poot_runtime_common::BufferRole::Output, elems * 4),
        }
    }

    /// i32 counterpart of [`Context::write_f32`]'s writable upload (for a `Slice<i32>`-typed slot,
    /// e.g. a paged decode's slot map). `pub(crate)`: no production caller survives Card 546b's
    /// `GpuExecutor` deletion; kept for the runtime-level storage-check tests (`src/tests.rs`).
    #[cfg(test)]
    pub(crate) fn upload_i32_writable(&self, data: &[i32]) -> DeviceBuffer {
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dev_slot_i32"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        self.record_upload_transfer(std::mem::size_of_val(data));
        DeviceBuffer {
            buf,
            elem_count: data.len() as u32,
            id: next_buffer_id(),
            storage: BufferStorage::i32(),
            _guard: self.tag_alloc(
                poot_runtime_common::BufferRole::Input,
                std::mem::size_of_val(data),
            ),
        }
    }

    /// Overwrite an existing device buffer's contents in place: no new `wgpu::Buffer`, just a queued
    /// `write_buffer` (ordered before any command buffer submitted afterward on this queue). `buf` must
    /// have been created with `COPY_DST` (see `upload_u32_writable`); writing into a plain
    /// [`Context::upload_f32`] buffer panics in wgpu validation. `data.len()` must equal `buf.elem_count`
    /// (the caller size-guards this; see `bind_resident`'s per-slot-kind numel check).
    ///
    /// A release check on `buf`'s storage (card 527, R471-009): a caller that legitimately changes what
    /// a reused buffer holds (a slot cache hit whose lane flips) retags it first with
    /// [`DeviceBuffer::with_storage`], so this only ever refuses a genuine mismatch.
    pub fn write_f32(&self, buf: &DeviceBuffer, data: &[f32]) -> Result<(), RuntimeError> {
        checked_storage("write_f32", buf, BufferStorage::f32())?;
        debug_assert_eq!(
            buf.elem_count as usize,
            data.len(),
            "write_f32: buffer/data length mismatch"
        );
        self.queue
            .write_buffer(&buf.buf, 0, bytemuck::cast_slice(data));
        self.record_upload_transfer(std::mem::size_of_val(data));
        Ok(())
    }

    /// u32 counterpart of [`Context::write_f32`]. Same storage check, against the generic word-buffer
    /// tag every `upload_u32`/`upload_u32_writable` caller uses (card 527): dims metadata, E4M3FN packed
    /// rows and packed-quant payloads all share it, so there is no single dtype to name. `pub(crate)`:
    /// no production caller survives Card 546b's `GpuExecutor` deletion; kept for the runtime-level
    /// `submit_cached` fault re-check test (`src/tests.rs`).
    #[cfg(test)]
    pub(crate) fn write_u32(&self, buf: &DeviceBuffer, data: &[u32]) -> Result<(), RuntimeError> {
        checked_storage(
            "write_u32",
            buf,
            BufferStorage::dense(ElementKind::I32, LogicalDType::RawBytes),
        )?;
        debug_assert_eq!(
            buf.elem_count as usize,
            data.len(),
            "write_u32: buffer/data length mismatch"
        );
        self.queue
            .write_buffer(&buf.buf, 0, bytemuck::cast_slice(data));
        self.record_upload_transfer(std::mem::size_of_val(data));
        Ok(())
    }

    /// Submit and complete this queue's pending uploads without encoding compute work or checking
    /// pending kernel faults. Polling alone does not submit queued writes. Test-only (card 546b):
    /// `GpuExecutor::prepare_decode_cache`, the one production caller, is gone with the rest of the
    /// pre-contract executor.
    #[cfg(test)]
    pub(crate) fn flush_uploads_and_wait(&self) -> Result<(), RuntimeError> {
        self.native_submits.set(self.native_submits.get() + 1);
        self.record_submit(poot_runtime_common::CallPurpose::Upload);
        self.queue.submit([]);
        self.record_wait(poot_runtime_common::CallPurpose::Upload);
        self.poll_wait()
    }

    /// Block until every submitted GPU command has completed, draining the queue (card 158). The
    /// resident prefill executor calls this after each batched `submit_cached` flush so the submitted
    /// work retires and its transient per-dispatch resources (length buffers, bind groups held by the
    /// flushed `CachedDispatch`es) are freed, bounding in-flight submissions and live objects over a very
    /// large prefill (~1e5 dispatches). Nothing is read back.
    pub fn poll_wait(&self) -> Result<(), RuntimeError> {
        self.waits.set(self.waits.get() + 1);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RuntimeError::Readback(e.to_string()))?;
        Ok(())
    }

    /// A zero-filled, writable, readable buffer of `elems` native elements of `storage`, charged to
    /// `role` (Card 546a/547a: the storage-typed, role-typed allocation the executor contract's memory
    /// service needs, instead of one allocator per dtype).
    pub fn alloc_storage(
        &self,
        role: poot_runtime_common::BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> DeviceBuffer {
        let bytes = (elems.max(1) * storage.element().byte_width()).div_ceil(4) * 4;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{role:?}")),
            size: bytes as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.buffer_allocs.set(self.buffer_allocs.get() + 1);
        DeviceBuffer {
            buf,
            elem_count: elems.max(1) as u32,
            id: next_buffer_id(),
            storage,
            _guard: self.tag_alloc(role, bytes),
        }
    }

    /// Overwrite `len` bytes of `buf` from byte `offset` (a whole number of its elements, padded to a
    /// word; Card 546a), in chunks of at most [`WRITE_CHUNK_BYTES`], each one wgpu staging allocation:
    /// `fill(range, out)` gets consecutive destination ranges covering `offset..offset + len`, at most
    /// 1 MiB at a time, so no host copy of the whole exists. The device writes whole words: `offset` and
    /// `len` must be multiples of four, except that a write reaching the end of the buffer may end
    /// mid-word (its last word is zero-padded). After [`PENDING_STAGING_BYTES`] of staging a write submits
    /// an empty command buffer, which starts the queued transfers and lets wgpu release their staging, so
    /// a large upload holds a bounded amount of it. Asynchronous: ordered before the next submitted work
    /// by the queue.
    pub fn write_with(
        &self,
        buf: &DeviceBuffer,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), RuntimeError> {
        checked_write_range("write_with", buf, offset, len, 1)?;
        let capacity = buf.elem_count as usize * buf.storage().element().byte_width();
        let reaches_end = offset + len == capacity;
        if !offset.is_multiple_of(4) || (!len.is_multiple_of(4) && !reaches_end) {
            return Err(RuntimeError::Misaligned {
                op: "write_with",
                offset,
                len,
            });
        }
        let mut scratch = vec![0u8; SCRATCH_BYTES.min(len)];
        let mut pending = 0;
        let mut done = 0;
        while done < len {
            let chunk = WRITE_CHUNK_BYTES.min(len - done);
            let padded = chunk.div_ceil(4) * 4;
            let size = wgpu::BufferSize::new(padded as u64).expect("a chunk is non-empty");
            let mut view = self
                .queue
                .write_buffer_with(&buf.buf, (offset + done) as wgpu::BufferAddress, size)
                .ok_or(RuntimeError::Misaligned {
                    op: "write_with",
                    offset: offset + done,
                    len: padded,
                })?;
            // wgpu's staging view is write-only (it may be write-combined memory): the caller's encoder
            // fills a cache-sized scratch, and each scratch run is copied into the view.
            let mut at = 0;
            while at < chunk {
                let run = SCRATCH_BYTES.min(chunk - at);
                let run_range = offset + done + at..offset + done + at + run;
                fill(run_range, &mut scratch[..run]);
                view.slice(at..at + run).copy_from_slice(&scratch[..run]);
                at += run;
            }
            view.slice(chunk..padded).fill(0);
            drop(view);
            done += chunk;
            pending += padded;
            if pending >= PENDING_STAGING_BYTES && done < len {
                // Not a compute submission: it carries no work, so no submit counter moves.
                self.queue.submit([]);
                pending = 0;
            }
        }
        self.record_upload_transfer(len);
        Ok(())
    }
}

/// The largest range one [`Context::write_with`] chunk covers: one staging allocation.
pub const WRITE_CHUNK_BYTES: usize = 64 << 20;

/// The bytes the caller's `fill` is handed at a time: a scratch that stays in cache between the encoder's
/// writes and the copy into wgpu's write-only staging view.
const SCRATCH_BYTES: usize = 1 << 20;

/// Staging a [`Context::write_with`] may queue before it submits to release it.
const PENDING_STAGING_BYTES: usize = 256 << 20;
