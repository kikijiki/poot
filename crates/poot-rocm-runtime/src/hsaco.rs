use crate::*;

/// Walk the HSACO ELF and return (name, file offset) for every `.kd` (kernel descriptor) symbol.
/// Used by `load_hsaco` to patch `kernel_code_properties` bit 3 (ENABLE_SGPR_KERNARG_SEGMENT_PTR) of
/// every kernel before the runtime copies the HSACO bytes.
pub(crate) fn find_kd_offsets(hsaco: &[u8]) -> Vec<(String, usize)> {
    let Ok(elf): Result<ElfFile<object::elf::FileHeader64<LittleEndian>, &'_ [u8]>, _> =
        ElfFile::parse(hsaco)
    else {
        return Vec::new();
    };
    elf.symbols()
        .filter_map(|s| {
            let name = s.name().ok()?.to_string();
            if !name.ends_with(".kd") {
                return None;
            }
            let section_idx = s.section_index()?;
            let section = elf.section_by_index(section_idx).ok()?;
            let (off, _len) = section.file_range()?;
            Some((name, off as usize))
        })
        .collect()
}

pub(crate) fn kernel_symbol_matches(discovered: &str, requested: &str) -> bool {
    discovered == requested
        || discovered.strip_suffix(".kd") == Some(requested)
        || requested.strip_suffix(".kd") == Some(discovered)
}

/// A loaded HSACO + the kernel handle needed to dispatch it. The executable owns the runtime-allocated
/// kernel_object handle and the loaded code object; drop destroys the executable, unless a timed-out
/// wait on the creating runtime means a hung packet may still run its code.
pub struct HsacoModule {
    /// The HSA executable that owns the loaded code object + kernel_object handle.
    pub(crate) exec: bindings::hsa_executable_t,
    /// Runtime-allocated `kernel_object` handle (from `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT`).
    /// Goes in the AQL packet's `kernel_object` field.
    pub kernel_object: u64,
    /// Kernarg buffer size in bytes (from the kernel descriptor).
    pub kernarg_size: u32,
    /// Group segment (LDS) size in bytes (from the kernel descriptor).
    pub group_segment_size: u32,
    /// Private segment (scratch) size in bytes per work-item (from the kernel descriptor).
    pub private_segment_size: u32,
    /// Exact KERNEL symbol selected while loading this single-kernel module.
    pub(crate) kernel_symbol: String,
    /// The originating initialized HSA runtime and executable-destruction authority.
    pub(crate) owner: Arc<dyn ResourceOwner>,
    /// Set once a timed-out submission of the originating runtime may still run this code object.
    pub(crate) poison: DevicePoison,
    /// The argument schema copied from the [`poot_runtime_common::CompiledKernel`] `load_hsaco` loaded
    /// this module from (card 608): `lookup_kernel` carries it into
    /// [`KernelHandle`], the only place this crate can still check a dispatch's bound buffers against it,
    /// since a ROCm kernarg is one opaque byte blob by the time it reaches [`RocmContext::dispatch`].
    pub(crate) args: Arc<[poot_runtime_common::ArgSchema]>,
}

impl std::fmt::Debug for HsacoModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HsacoModule")
            .field("exec", &format_args!("0x{:x}", self.exec.handle))
            .field("kernel_object", &format_args!("0x{:x}", self.kernel_object))
            .field("kernarg_size", &self.kernarg_size)
            .field("group_segment_size", &self.group_segment_size)
            .field("private_segment_size", &self.private_segment_size)
            .field("kernel_symbol", &self.kernel_symbol)
            .finish()
    }
}

impl Drop for HsacoModule {
    fn drop(&mut self) {
        if self.exec.handle != 0 && !self.poison.is_poisoned() {
            self.owner.destroy_executable(self.exec);
        }
    }
}

/// Resources acquired while constructing one [`HsacoModule`]. Until `finish` moves the executable
/// into the returned module, every early return drops this value and unwinds all handles: no packet
/// can reference an executable that was never returned.
pub(crate) struct ModuleConstruction {
    pub(crate) owner: Arc<dyn ResourceOwner>,
    pub(crate) poison: DevicePoison,
    pub(crate) reader: Option<bindings::hsa_code_object_reader_t>,
    pub(crate) executable: Option<bindings::hsa_executable_t>,
}

impl ModuleConstruction {
    pub(crate) fn new(owner: Arc<dyn ResourceOwner>, poison: DevicePoison) -> Self {
        Self {
            owner,
            poison,
            reader: None,
            executable: None,
        }
    }

    pub(crate) fn finish(
        mut self,
        kernel_object: u64,
        kernarg_size: u32,
        group_segment_size: u32,
        private_segment_size: u32,
        kernel_symbol: String,
        args: Arc<[poot_runtime_common::ArgSchema]>,
    ) -> HsacoModule {
        let exec = self
            .executable
            .take()
            .expect("module construction finished without an executable");
        HsacoModule {
            exec,
            kernel_object,
            kernarg_size,
            group_segment_size,
            private_segment_size,
            kernel_symbol,
            owner: Arc::clone(&self.owner),
            poison: self.poison.clone(),
            args,
        }
    }
}

impl Drop for ModuleConstruction {
    fn drop(&mut self) {
        if let Some(executable) = self.executable.take() {
            self.owner.destroy_executable(executable);
        }
        if let Some(reader) = self.reader.take() {
            self.owner.destroy_reader(reader);
        }
    }
}

/// Context passed to `for_each_cb` (the `hsa_executable_iterate_symbols` callback), carrying the
/// `hsa_executable_symbol_get_info` function pointer used to query each symbol's kind and name.
pub(crate) struct LookupCtx {
    pub(crate) funcs: *const ffi::Funcs,
    pub(crate) data: *mut c_void,
}

/// Iterate-symbols callback. Records the first KERNEL-kind symbol name (and the first with a `.kd`
/// suffix if present). One kernel is loaded per HSACO, so the first match is the only one.
///
/// # Safety
///
/// `data` must point to a live [`LookupCtx`] whose `funcs` table is live and whose `data` points to
/// an `(Option<String>, Option<String>)` nothing else accesses during the call.
unsafe extern "C" fn for_each_cb(
    _exec: bindings::hsa_executable_t,
    sym: bindings::hsa_executable_symbol_t,
    data: *mut c_void,
) -> bindings::hsa_status_t {
    // SAFETY: `data` is the live `LookupCtx` `load_hsaco` passed (function contract).
    let ctx = unsafe { &*(data as *const LookupCtx) };
    let mut kind: u32 = 0;
    // SAFETY: `ctx.funcs` is live; `HSA_EXECUTABLE_SYMBOL_INFO_TYPE` writes a 4-byte enum into `kind`.
    let s = unsafe {
        ((*ctx.funcs).hsa_executable_symbol_get_info)(
            sym,
            bindings::HSA_EXECUTABLE_SYMBOL_INFO_TYPE,
            &mut kind as *mut u32 as *mut c_void,
        )
    };
    if s != bindings::HSA_STATUS_SUCCESS {
        return s;
    }
    if kind == bindings::HSA_SYMBOL_KIND_KERNEL {
        tracing::trace!("for_each_cb: found KERNEL, querying name");
        let mut name_length: u32 = 0;
        // SAFETY: `ctx.funcs` is live; NAME_LENGTH writes a `uint32_t` into `name_length`.
        let s1 = unsafe {
            ((*ctx.funcs).hsa_executable_symbol_get_info)(
                sym,
                bindings::HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH,
                &mut name_length as *mut u32 as *mut c_void,
            )
        };
        if s1 != bindings::HSA_STATUS_SUCCESS {
            return s1;
        }
        // The name is `name_length` bytes; the extra zero byte terminates it.
        let mut name_buf = vec![0u8; name_length as usize + 1];
        // SAFETY: `ctx.funcs` is live; NAME writes `name_length` bytes into `name_buf`, which holds one
        // more.
        let s2 = unsafe {
            ((*ctx.funcs).hsa_executable_symbol_get_info)(
                sym,
                bindings::HSA_EXECUTABLE_SYMBOL_INFO_NAME,
                name_buf.as_mut_ptr() as *mut c_void,
            )
        };
        if s2 == bindings::HSA_STATUS_SUCCESS {
            let nul = name_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_buf.len());
            let name = String::from_utf8_lossy(&name_buf[..nul]).into_owned();
            tracing::trace!(?name, "for_each_cb: symbol name");
            // SAFETY: `ctx.data` points to the name pair only this callback accesses (function contract).
            let state = unsafe { &mut *(ctx.data as *mut (Option<String>, Option<String>)) };
            if state.0.is_none() {
                state.0 = Some(name.clone());
            }
            if name.ends_with(".kd") && state.1.is_none() {
                state.1 = Some(name);
            }
        } else {
            tracing::trace!(
                status = format_args!("0x{s2:x}"),
                "for_each_cb: name query failed"
            );
        }
    } else {
        tracing::trace!(kind, "for_each_cb: skipping non-KERNEL symbol");
    }
    bindings::HSA_STATUS_SUCCESS
}

impl RocmContext {
    /// Load a poot-emitted HSACO blob through the standard HSA Foundation API:
    /// `hsa_executable_load_agent_code_object` with a `hsa_code_object_reader_t` (the V4 code-object
    /// flow; the legacy `hsa_executable_load_code_object` does not accept V4 readers).
    ///
    /// The kernel symbol in V4 code objects is `<kernel>.kd` (e.g. `add.kd`); the runtime allocates the
    /// `kernel_object` handle when the code object loads, and it is queried via
    /// `hsa_executable_symbol_get_info(..., KERNEL_OBJECT, ...)`.
    ///
    /// The returned `HsacoModule` owns the `hsa_executable_t`; `Drop` destroys it.
    ///
    /// `kernel` must be a compiler-produced [`poot_runtime_common::CompiledKernel`] (card 608): the only
    /// safe way to reach this call with kernel bytes at all, since a `CompiledKernel`'s only constructor
    /// is `unsafe` and `poot-codegen` is its one caller (see the crate doc for the compile-fail proof).
    /// Returns [`RocmError::WrongKernelTarget`] if `kernel` was compiled for another backend.
    pub fn load_hsaco(
        &self,
        kernel: &poot_runtime_common::CompiledKernel,
    ) -> Result<HsacoModule, RocmError> {
        let poot_runtime_common::KernelCode::Hsaco(hsaco) = kernel.code() else {
            return Err(RocmError::WrongKernelTarget {
                actual: kernel.target(),
            });
        };
        let mut construction =
            ModuleConstruction::new(self.resource_owner(), self.timeouts.device_poison());
        // 0. Ensure `kernel_code_properties.ENABLE_SGPR_KERNARG_SEGMENT_PTR` (bit 3) is set for every
        // kernel. LLVM 22 for gfx1151/AMDHSA already sets it (observed `kernel_code_properties = 0x0408`:
        // bit 3 = KERNARG, bit 10 = WAVEFRONT_SIZE32), so this patch is a safety net for compiler drift.
        //
        // Kernel descriptor layout (64-byte, Code Object V3, AMDHSAKernelDescriptor.h):
        //   +48: compute_pgm_rsrc1 (u32)
        //   +52: compute_pgm_rsrc2 (u32)  (TGID_X_EN at bit 7, USER_SGPR_COUNT at bits 1-5)
        //   +56: kernel_code_properties (u16)  (ENABLE_SGPR_KERNARG_SEGMENT_PTR at bit 3)
        //
        // kernel_code_properties is at +56, not +52: patching +52 corrupts
        // compute_pgm_rsrc2.USER_SGPR_COUNT, so workgroup_id_x reads as 0 in multi-workgroup dispatches.
        // LLVM sets TGID_X_EN itself; do not touch it.
        let mut patched = hsaco.to_vec();
        let kd_offsets = find_kd_offsets(&patched);
        if kd_offsets.is_empty() {
            tracing::warn!("no .kd symbols found for properties-patch pass; dispatch may fail");
        } else {
            for (_name, kd_off) in &kd_offsets {
                let props_off = kd_off + 56; // kernel_code_properties, NOT compute_pgm_rsrc2 (+52)
                if patched.len() >= props_off + 2 {
                    let props = u16::from_le_bytes([patched[props_off], patched[props_off + 1]]);
                    const ENABLE_SGPR_KERNARG_SEGMENT_PTR: u16 = 1 << 3;
                    let new_props = props | ENABLE_SGPR_KERNARG_SEGMENT_PTR;
                    patched[props_off] = (new_props & 0xff) as u8;
                    patched[props_off + 1] = (new_props >> 8) as u8;
                    tracing::trace!(
                        kd_off = format_args!("0x{kd_off:x}"),
                        props = format_args!("{props:#06x}"),
                        new_props = format_args!("{new_props:#06x}"),
                        "patched kd kernel_code_properties",
                    );
                }
            }
        }
        // 1. Wrap the HSACO bytes in a code-object reader.
        let mut reader = bindings::hsa_code_object_reader_t { handle: 0 };
        // SAFETY: `patched` outlives the reader, which `construction` destroys before returning.
        unsafe {
            check((self.hsa.funcs.hsa_code_object_reader_create_from_memory)(
                patched.as_ptr() as *const c_void,
                patched.len(),
                &mut reader,
            ))?;
        }
        construction.reader = Some(reader);
        // 2. Create an executable (unfrozen) on the GPU agent.
        let mut exec = bindings::hsa_executable_t { handle: 0 };
        // SAFETY: null options are allowed; `exec` is a valid out-pointer.
        unsafe {
            check((self.hsa.funcs.hsa_executable_create_alt)(
                bindings::HSA_PROFILE_FULL,
                bindings::HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                std::ptr::null(),
                &mut exec,
            ))?;
        }
        construction.executable = Some(exec);
        // 3. Load the code object. The runtime allocates its own kernel_object handle for the kernel
        // descriptor; queried below.
        // SAFETY: `exec` and `reader` are live handles just created; null options and a null
        // loaded-code-object out-pointer are allowed.
        unsafe {
            check((self.hsa.funcs.hsa_executable_load_agent_code_object)(
                exec,
                self.gpu_agent,
                reader,
                std::ptr::null(),
                std::ptr::null_mut(),
            ))?;
        }
        // 4. Freeze the executable so further lookups + dispatches are valid.
        // SAFETY: `exec` is live; null options are allowed.
        unsafe {
            check((self.hsa.funcs.hsa_executable_freeze)(
                exec,
                std::ptr::null(),
            ))?;
        }
        // 5. Discover the kernel symbol name: one kernel per module, so take the first KERNEL-kind symbol
        // (preferring the `.kd`-suffixed one).
        let mut state: (Option<String>, Option<String>) = (None, None);
        let ctx = LookupCtx {
            funcs: &self.hsa.funcs,
            data: &mut state as *mut _ as *mut c_void,
        };
        let cb: bindings::hsa_executable_iterate_symbols_cb_t = Some(for_each_cb);
        // SAFETY: the iteration is synchronous, and `ctx` (with the table and `state` it points to)
        // outlives it, meeting `for_each_cb`'s contract.
        unsafe {
            check((self.hsa.funcs.hsa_executable_iterate_symbols)(
                exec,
                cb,
                &ctx as *const _ as *mut c_void,
            ))?;
        }
        let sym_name = state
            .1
            .or(state.0)
            .ok_or_else(|| RocmError::Hsa("no KERNEL symbol found in HSACO executable".into()))?;
        tracing::trace!(?sym_name, "load_hsaco: found kernel symbol");
        // 6. Look up the kernel symbol by name.
        let mut sym = bindings::hsa_executable_symbol_t { handle: 0 };
        let agent = self.gpu_agent;
        let cname = std::ffi::CString::new(sym_name.as_bytes()).unwrap();
        // SAFETY: `exec` is live and frozen; `cname` is NUL-terminated; `agent` and `sym` are valid
        // for the call.
        unsafe {
            check((self.hsa.funcs.hsa_executable_get_symbol_by_name)(
                exec,
                cname.as_ptr(),
                &agent,
                &mut sym,
            ))?;
        }
        // 7. Query the kernel_object handle (runtime-allocated; what we put in AQL packets).
        let mut kernel_object: u64 = 0;
        let mut kernarg_size: u32 = 0;
        let mut group_segment_size: u32 = 0;
        let mut private_segment_size: u32 = 0;
        // SAFETY: `sym` is a live kernel symbol; KERNEL_OBJECT writes a `uint64_t` and the three
        // segment-size attributes a `uint32_t` each, into locals of those types.
        unsafe {
            check((self.hsa.funcs.hsa_executable_symbol_get_info)(
                sym,
                bindings::HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                &mut kernel_object as *mut u64 as *mut c_void,
            ))?;
            check((self.hsa.funcs.hsa_executable_symbol_get_info)(
                sym,
                bindings::HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                &mut kernarg_size as *mut u32 as *mut c_void,
            ))?;
            check((self.hsa.funcs.hsa_executable_symbol_get_info)(
                sym,
                bindings::HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                &mut group_segment_size as *mut u32 as *mut c_void,
            ))?;
            check((self.hsa.funcs.hsa_executable_symbol_get_info)(
                sym,
                bindings::HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                &mut private_segment_size as *mut u32 as *mut c_void,
            ))?;
        }
        tracing::info!(
            "load_hsaco: {} bytes, kernel={} kernel_object=0x{:x} kernarg={} group={} private={}",
            hsaco.len(),
            sym_name,
            kernel_object,
            kernarg_size,
            group_segment_size,
            private_segment_size,
        );
        // 8. Transfer the executable to the module. Dropping `construction` destroys the reader; the
        // executable is owned by the returned module.
        Ok(construction.finish(
            kernel_object,
            kernarg_size,
            group_segment_size,
            private_segment_size,
            sym_name,
            Arc::from(kernel.args()),
        ))
    }

    /// Look up a kernel symbol in a loaded module. `HsacoModule` already holds the runtime-allocated
    /// `kernel_object` handle (`HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT`), so this is constant-time.
    pub fn lookup_kernel(
        &self,
        module: &HsacoModule,
        kernel: &str,
    ) -> Result<KernelHandle, RocmError> {
        if !kernel_symbol_matches(&module.kernel_symbol, kernel) {
            return Err(RocmError::Hsa(format!(
                "HSACO contains kernel symbol {:?}, not requested symbol {kernel:?}",
                module.kernel_symbol
            )));
        }
        // The runtime allocated `kernel_object` for the exact symbol at load; after the identity check
        // above, pass it through to the AQL packet.
        Ok(KernelHandle {
            kernel_object: module.kernel_object,
            kernarg_size: module.kernarg_size,
            group_segment_size: module.group_segment_size,
            private_segment_size: module.private_segment_size,
            args: Arc::clone(&module.args),
        })
    }

    /// Shared bounded completion wait for `dispatch`, `replay_graph_batched` and `dma_copy`. Polls
    /// `hsa_signal_wait_scacquire` in short chunks (not one `timeout_hint = u64::MAX` wait) so a lost
    /// completion signal surfaces as `QueueWaitTimeout` instead of an infinite host spin.
    ///
    /// The per-attempt `timeout_hint` is 1ms, not 1s: on gfx1151 this stack does not return from
    /// `hsa_signal_wait_scacquire` when the signal drops mid-wait - it sleeps the full hint
    /// (measured ~1.00-1.08s per wait with a 1s hint, for both `BLOCKED` and `ACTIVE`). Captured
    /// olmo2 decode then measured ~2.3s/token (replay wait + argmax wait) vs ~0.44 tok/s reported.
    /// A 1ms hint bounds the wake latency; the wall-clock loop below still enforces
    /// [`RocmContext::wait_timeout`] across attempts.
    ///
    /// The signal is released only on proven completion: on success (signal value below 1) it is
    /// destroyed and `held` is handed back, so the caller may read and then drop it. On timeout the
    /// signal and `held` move to the context's leak list, because the packet or copy may still use
    /// them, and the runtime is poisoned, so no allocation or code object of it is released (see
    /// [`DevicePoison`]). Returns `Err(RocmError::QueueWaitTimeout(effective_timeout, tag))`. The
    /// bound is the typed [`RocmContextOptions::wait_timeout`] this context was constructed with
    /// (Card 548).
    pub(crate) fn wait_completion_bounded(
        &self,
        completion: bindings::hsa_signal_t,
        tag: &'static str,
        held: Held,
    ) -> Result<Held, RocmError> {
        let timeout = self.wait_timeout.as_secs();
        let start = std::time::Instant::now();
        loop {
            // SAFETY: `completion` is a live signal this context created and has not destroyed.
            let value = unsafe {
                (self.hsa.funcs.hsa_signal_wait_scacquire)(
                    completion,
                    bindings::HSA_SIGNAL_CONDITION_LT,
                    1,
                    // 1ms per-attempt timeout_hint: this stack sleeps the full hint before
                    // re-reading the signal, so a long hint is a fixed latency per dispatch.
                    1_000_000,
                    bindings::HSA_WAIT_STATE_BLOCKED,
                )
            };
            if value < 1 {
                // SAFETY: the packet or copy has completed, so nothing uses `completion` any more; it
                // is destroyed once.
                unsafe {
                    let _ = (self.hsa.funcs.hsa_signal_destroy)(completion);
                }
                return Ok(held);
            }
            if start.elapsed().as_secs() >= timeout {
                self.timeouts.abandon(Abandoned {
                    signal: completion,
                    held,
                });
                return Err(RocmError::QueueWaitTimeout(timeout, tag));
            }
        }
    }

    /// Dispatch a single kernel and wait on its completion signal. The kernarg buffer must already
    /// hold the kernel's args (the `(ptr, i64 len, ptr, i64 len, ...)` sequence for slice-bearing
    /// kernels). `grid_x * grid_y * grid_z` work-items in `block_x * block_y * block_z` blocks (the
    /// workgroup size; 64 or 128 for gfx1151).
    ///
    /// The packet goes through the same publisher as a replayed graph (a one-packet batch), the one
    /// place that writes the body with an `INVALID` header and release-stores the header last. If the
    /// wait times out the context is poisoned, and the kernarg allocation, the data allocations it
    /// points at and the kernel's code object stay alive whatever the caller does with its own
    /// handles (see [`Self::wait_completion_bounded`]).
    ///
    /// `elements` is `kernarg`'s packed buffers' native element kinds, in kernarg slot order (card 608,
    /// SC-002): checked against `kernel.args()` before any submission, since a
    /// ROCm kernarg is one opaque byte blob by this point - the caller that flattened it into `kernarg`
    /// (e.g. `poot-rocm-gpu`'s `build_flat_kernarg`) is the one place that still knows what each slot is.
    pub fn dispatch(
        &self,
        kernel: &KernelHandle,
        kernarg: &RocmBuffer,
        elements: &[poot_target::ElementKind],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), RocmError> {
        validate_kernarg_buffer("dispatch", kernel, kernarg)?;
        poot_runtime_common::check_element_schema(kernel.args(), elements)?;
        self.submit_and_wait(
            &[(kernel.clone(), kernarg.clone(), grid, block)],
            "dispatch",
        )?;
        Ok(())
    }

    /// Dispatch a batch of kernels with a single doorbell ring per chunk. All AQL packets are written to
    /// the ring, then the doorbell rung once; the CP processes them in order (BARRIER bit set). Only the
    /// last packet has a completion signal; earlier ones use `HSA_SIGNAL_INVALID`.
    ///
    /// Each entry is `(kernel, kernarg, grid, block)`. Every owned kernarg buffer must already hold the
    /// kernel's args and have the exact kernel-declared byte capacity. `elements[i]` is `dispatches[i]`'s
    /// packed buffers' native element kinds (see [`Self::dispatch`]); checked against `dispatches[i].0`'s
    /// schema before any submission (card 608, SC-002).
    ///
    /// When the batch exceeds half the queue size it is dispatched in chunks, spinning on `read_index`
    /// until each fits. Returns the number of physical AQL batch submissions (doorbell rings), which
    /// exceeds one when `dispatches` does not fit in a single chunk.
    #[allow(clippy::type_complexity)]
    pub fn replay_graph_batched(
        &self,
        dispatches: &[(KernelHandle, RocmBuffer, (u32, u32, u32), (u32, u32, u32))],
        elements: &[&[poot_target::ElementKind]],
    ) -> Result<usize, RocmError> {
        let n = dispatches.len();
        if n == 0 {
            return Ok(0);
        }
        assert_eq!(
            elements.len(),
            n,
            "replay_graph_batched: dispatches and elements must be reported pairwise (caller bug)"
        );
        for ((kernel, kernarg, _, _), &elems) in dispatches.iter().zip(elements) {
            validate_kernarg_buffer("replay_graph_batched", kernel, kernarg)?;
            poot_runtime_common::check_element_schema(kernel.args(), elems)?;
        }
        self.submit_and_wait(dispatches, "replay_graph_batched")
    }

    /// Publish `dispatches` through the queue's publication contract and wait on the last packet's
    /// completion signal via `wait_completion_bounded`, so a lost signal surfaces as
    /// `QueueWaitTimeout` instead of an infinite host spin (the card-188 batched-serve hang, likely the
    /// card-186 gemma4-MoE hang). Returns the number of physical submissions.
    ///
    /// A poisoned context is refused before the queue is touched. Once the publisher has reserved ring
    /// slots, a failure (the queue-space wait timing out) can leave packets in flight and a slot
    /// reserved but never published, so it leaks the completion signal like a timed-out completion
    /// wait and poisons the context.
    #[allow(clippy::type_complexity)]
    fn submit_and_wait(
        &self,
        dispatches: &[(KernelHandle, RocmBuffer, (u32, u32, u32), (u32, u32, u32))],
        tag: &'static str,
    ) -> Result<usize, RocmError> {
        self.ensure_not_poisoned(tag)?;

        // Create a completion signal for the last packet, initialised to 1; the packet decrements it.
        let mut completion: bindings::hsa_signal_t = bindings::hsa_signal_t { handle: 0 };
        // SAFETY: zero consumers with a null list is allowed; `completion` is a valid out-pointer.
        unsafe {
            check((self.hsa.funcs.hsa_signal_create)(
                1,
                0,
                std::ptr::null(),
                &mut completion,
            ))?;
        }

        let q = self.queue_layout();
        let queue_size = q.size as u64;
        let ring = q.base_address as *mut bindings::hsa_kernel_dispatch_packet_t;

        // AQL header: KERNEL_DISPATCH + BARRIER + SCACQUIRE/RELEASE = AGENT. Without BARRIER the kernel
        // can dispatch out of order relative to other packets on the queue, which on gfx1151 / ROCm
        // 7.2.3 shows as workgroup_size being ignored (only `grid_size_x` work-items launch).
        let header: u16 = 2u16 | (1u16 << 8) | (1u16 << 9) | (1u16 << 11);
        let mut publisher = LiveAqlChunkSink {
            context: self,
            doorbell: q.doorbell_signal,
            writer: AqlRingWriter {
                dispatches,
                completion,
                ring,
                header,
            },
        };
        let physical_submissions = match self.queue_publication.publish_chunks(
            dispatches.len(),
            queue_size,
            &mut publisher,
        ) {
            Ok(submissions) => submissions,
            Err(error) => {
                self.timeouts.abandon(Abandoned {
                    signal: completion,
                    held: Held::default(),
                });
                return Err(error);
            }
        };

        self.wait_completion_bounded(completion, tag, Held::default())?;
        Ok(physical_submissions)
    }

    pub fn queue_role(&self) -> QueueRole {
        self.queue_role
    }

    pub fn queue_ring_provenance(&self) -> QueueRingProvenance {
        self.queue_ring_provenance
    }

    /// Publication ordering retained by this context's sole HSA queue.
    pub fn queue_publication_contract(&self) -> QueuePublicationContract {
        self.queue_publication.contract()
    }

    /// Device-ring receipt provenance when this context was opened for Card 333 PCIe verification.
    pub fn device_ring_receipt_provenance(&self) -> Option<&DeviceRingReceiptProvenance> {
        self.device_ring_receipt.as_ref()
    }
}

/// Per-launch kernel metadata queried from the executable symbol.
///
/// Fields are private (card 608): the only constructor is [`RocmContext::lookup_kernel`], which queries
/// them from a genuinely loaded [`HsacoModule`], so safe code cannot forge a `kernel_object` that points
/// an AQL packet at an arbitrary address. It also carries the argument schema `load_hsaco` copied from the
/// `CompiledKernel` it loaded (card 608), so `RocmContext::dispatch`/
/// `replay_graph_batched` can check a dispatch's bound buffers before submission even though a ROCm
/// kernarg is one opaque byte blob by then.
///
/// ```compile_fail,E0451
/// fn forge() -> poot_rocm_runtime::KernelHandle {
///     poot_rocm_runtime::KernelHandle {
///         kernel_object: 0,
///         kernarg_size: 0,
///         group_segment_size: 0,
///         private_segment_size: 0,
///         args: std::sync::Arc::from([]),
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct KernelHandle {
    /// EXEC symbol handle. Goes in the AQL packet's `kernel_object` field.
    kernel_object: u64,
    /// Kernarg buffer size in bytes (must be allocated at least this large).
    kernarg_size: u32,
    /// Group segment (LDS) size in bytes (must be in the AQL packet's `group_segment_size`).
    group_segment_size: u32,
    /// Private segment (scratch) size in bytes per work-item.
    private_segment_size: u32,
    /// Copied from [`HsacoModule::args`] (see the struct doc).
    args: Arc<[poot_runtime_common::ArgSchema]>,
}

impl KernelHandle {
    /// A synthetic handle for tests that exercise AQL packet building or kernarg validation without a
    /// real HSA load. Not `pub`: production code only ever gets one from
    /// [`RocmContext::lookup_kernel`].
    #[cfg(test)]
    pub(crate) fn for_test(
        kernel_object: u64,
        kernarg_size: u32,
        group_segment_size: u32,
        private_segment_size: u32,
        args: &[poot_runtime_common::ArgSchema],
    ) -> Self {
        KernelHandle {
            kernel_object,
            kernarg_size,
            group_segment_size,
            private_segment_size,
            args: Arc::from(args),
        }
    }

    /// The AQL packet's `kernel_object` field.
    pub fn kernel_object(&self) -> u64 {
        self.kernel_object
    }

    /// The kernarg buffer's required byte capacity.
    pub fn kernarg_size(&self) -> u32 {
        self.kernarg_size
    }

    /// The AQL packet's `group_segment_size` field.
    pub fn group_segment_size(&self) -> u32 {
        self.group_segment_size
    }

    /// The AQL packet's `private_segment_size` field.
    pub fn private_segment_size(&self) -> u32 {
        self.private_segment_size
    }

    /// This kernel's argument schema, one entry per data buffer it binds, in binding order (card 608):
    /// copied from the `CompiledKernel` `load_hsaco` loaded it from.
    pub fn args(&self) -> &[poot_runtime_common::ArgSchema] {
        &self.args
    }
}

pub(crate) fn validate_kernarg_buffer(
    op: &'static str,
    kernel: &KernelHandle,
    kernarg: &RocmBuffer,
) -> Result<(), RocmError> {
    if kernarg.storage() != BufferStorage::raw_bytes() {
        return Err(RocmError::RepresentationMismatch {
            op,
            expected: BufferStorage::raw_bytes(),
            actual: kernarg.storage(),
        });
    }
    let expected = kernel.kernarg_size() as usize;
    if kernarg.byte_capacity() != expected {
        return Err(RocmError::KernargSizeMismatch {
            op,
            expected,
            actual: kernarg.byte_capacity(),
        });
    }
    Ok(())
}

/// Allocate an owned kernarg buffer of `size` bytes. It comes from the fine-grained kernarg-capable
/// pool, is freed when its last [`RocmBuffer`] handle drops, and is initialized through the context's
/// selected host-write or DMA strategy.
impl RocmContext {
    pub fn allocate_kernarg(&self, size: usize) -> Result<RocmBuffer, RocmError> {
        self.allocate_zeroed(
            "allocate_kernarg",
            size,
            BufferStorage::raw_bytes(),
            BufferRole::Meta,
        )
    }
}
