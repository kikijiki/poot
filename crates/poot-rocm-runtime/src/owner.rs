use crate::*;

/// Cleanup operations carried by every resource created from one initialized HSA runtime.
///
/// The trait object is narrow: graph execution goes through [`RocmContext`], while allocations and
/// modules retain only what their destructors need. `Send + Sync` records HSA's process-wide,
/// thread-safe cleanup contract and lets a resource follow the server executor to its engine thread.
///
/// Crate contract: each argument is a live handle this runtime created, released exactly once, by the
/// resource that owns it, after no queued packet can still use it.
pub(crate) trait ResourceOwner: Send + Sync {
    fn free_memory(&self, ptr: *mut c_void);
    fn destroy_reader(&self, reader: bindings::hsa_code_object_reader_t);
    fn destroy_executable(&self, executable: bindings::hsa_executable_t);
}

/// One successful `hsa_init` plus the loaded library and function table used by its resources.
/// `Arc<ResourceOwner>` references in allocations and modules keep it alive after a [`RocmContext`]
/// moves threads or drops, so its destructor is the single matching `hsa_shut_down` and runs only
/// after every owned HSA resource is destroyed.
pub(crate) struct HsaRuntimeOwner {
    pub(crate) hsa: Hsa,
    /// Live/peak bytes and allocation count by role (Card 547a): the one memory service's view of
    /// every allocation this runtime has made.
    pub(crate) memory: poot_runtime_common::MemoryCounters,
}

impl std::ops::Deref for HsaRuntimeOwner {
    type Target = Hsa;

    fn deref(&self) -> &Self::Target {
        &self.hsa
    }
}

impl ResourceOwner for HsaRuntimeOwner {
    fn free_memory(&self, ptr: *mut c_void) {
        // SAFETY: `ptr` is a live pool allocation of this runtime, freed once (trait contract).
        unsafe {
            let _ = (self.hsa.funcs.hsa_amd_memory_pool_free)(ptr);
        }
    }

    fn destroy_reader(&self, reader: bindings::hsa_code_object_reader_t) {
        // SAFETY: `reader` is a live code-object reader of this runtime, destroyed once (trait contract).
        unsafe {
            let _ = (self.hsa.funcs.hsa_code_object_reader_destroy)(reader);
        }
    }

    fn destroy_executable(&self, executable: bindings::hsa_executable_t) {
        // SAFETY: `executable` is a live executable of this runtime, destroyed once (trait contract).
        unsafe {
            let _ = (self.hsa.funcs.hsa_executable_destroy)(executable);
        }
    }
}

impl Drop for HsaRuntimeOwner {
    fn drop(&mut self) {
        // SAFETY: balances this owner's one successful `hsa_init`. Every resource of the runtime holds
        // an `Arc` of this owner, so none is left, and the library is still loaded (`hsa` drops after).
        unsafe {
            let _ = (self.hsa.funcs.hsa_shut_down)();
        }
    }
}
