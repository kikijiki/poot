use cudarc::driver::{DriverError, result, sys};

/// The driver calls behind every PTX handle's cleanup and graph replay, behind a trait so lifecycle tests can
/// record them. Crate contract: each handle argument is a live handle the calling [`PtxOwner`] or one of its
/// children created and has not yet destroyed, each destroy runs at most once per handle, and handle-taking
/// calls run with the owning context current (a `CurrentContextGuard`).
///
/// [`PtxOwner`]: crate::owner::PtxOwner
pub(crate) trait CudaDriver: Send + Sync {
    fn get_current(&self) -> Result<Option<sys::CUcontext>, DriverError>;
    fn set_current(&self, ctx: Option<sys::CUcontext>) -> Result<(), DriverError>;
    fn free(&self, ptr: sys::CUdeviceptr);
    fn unload_module(&self, module: sys::CUmodule);
    fn destroy_stream(&self, stream: sys::CUstream);
    fn release_primary(&self, device: sys::CUdevice);
    fn destroy_event(&self, event: sys::CUevent);
    fn destroy_graph_exec(&self, exec: sys::CUgraphExec);
    fn destroy_graph(&self, graph: sys::CUgraph);
    fn launch_graph(
        &self,
        exec: sys::CUgraphExec,
        stream: sys::CUstream,
    ) -> Result<(), DriverError>;
    fn synchronize_stream(&self, stream: sys::CUstream) -> Result<(), DriverError>;
}

pub(crate) struct DriverCleanup;

impl CudaDriver for DriverCleanup {
    fn get_current(&self) -> Result<Option<sys::CUcontext>, DriverError> {
        result::ctx::get_current()
    }

    fn set_current(&self, ctx: Option<sys::CUcontext>) -> Result<(), DriverError> {
        // SAFETY: `ctx` is null (unbind) or a context a caller observed current or retained, per the trait
        // contract.
        unsafe { result::ctx::set_current(ctx.unwrap_or(std::ptr::null_mut())) }
    }

    fn free(&self, ptr: sys::CUdeviceptr) {
        // SAFETY: `ptr` is a live allocation of the current context, freed once (trait contract).
        unsafe {
            let _ = result::free_sync(ptr);
        }
    }

    fn unload_module(&self, module: sys::CUmodule) {
        // SAFETY: `module` is a live module of the current context, unloaded once (trait contract).
        unsafe {
            let _ = result::module::unload(module);
        }
    }

    fn destroy_stream(&self, stream: sys::CUstream) {
        // SAFETY: `stream` is the owner's live stream, destroyed once (trait contract).
        unsafe {
            let _ = result::stream::destroy(stream);
        }
    }

    fn release_primary(&self, device: sys::CUdevice) {
        // SAFETY: balances the one primary-context retain the owner took on `device` (trait contract).
        unsafe {
            let _ = result::primary_ctx::release(device);
        }
    }

    fn destroy_event(&self, event: sys::CUevent) {
        // SAFETY: `event` is a live event of the current context, destroyed once (trait contract).
        unsafe {
            let _ = result::event::destroy(event);
        }
    }

    fn destroy_graph_exec(&self, exec: sys::CUgraphExec) {
        // SAFETY: `exec` is a live executable graph, destroyed once (trait contract).
        unsafe {
            let _ = sys::cuGraphExecDestroy(exec).result();
        }
    }

    fn destroy_graph(&self, graph: sys::CUgraph) {
        // SAFETY: `graph` is a live graph, destroyed once (trait contract).
        unsafe {
            let _ = sys::cuGraphDestroy(graph).result();
        }
    }

    fn launch_graph(
        &self,
        exec: sys::CUgraphExec,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        // SAFETY: `exec` and `stream` are live handles of the current context (trait contract), and the
        // executable graph owns every buffer it addresses.
        unsafe { sys::cuGraphLaunch(exec, stream) }.result()
    }

    fn synchronize_stream(&self, stream: sys::CUstream) -> Result<(), DriverError> {
        // SAFETY: `stream` is the owner's live stream (trait contract).
        unsafe { result::stream::synchronize(stream) }
    }
}
