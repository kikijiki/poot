use std::sync::{Arc, Mutex};

use cudarc::driver::{DriverError, sys};

use super::driver::CudaDriver;
use super::error::PtxError;

pub(crate) struct ContextActivationError {
    pub(crate) source: DriverError,
    pub(crate) restored_prior: Option<Option<sys::CUcontext>>,
}

fn current_before_owner_release(
    prior: Option<sys::CUcontext>,
    owner: sys::CUcontext,
) -> Option<sys::CUcontext> {
    if prior == Some(owner) { None } else { prior }
}

/// Temporarily makes one CUDA context current on this thread and restores the exact prior state. `None` is
/// significant: CUDA reports success when no context is current, and restoration must pass a null context to
/// unbind the temporary owner.
pub(crate) struct CurrentContextGuard {
    driver: Arc<dyn CudaDriver>,
    prior: Option<Option<sys::CUcontext>>,
}

impl CurrentContextGuard {
    pub(crate) fn activate(
        driver: &Arc<dyn CudaDriver>,
        owner: sys::CUcontext,
    ) -> Result<Self, ContextActivationError> {
        let prior = driver
            .get_current()
            .map_err(|source| ContextActivationError {
                source,
                restored_prior: None,
            })?;
        if let Err(source) = driver.set_current(Some(owner)) {
            let restored_prior = driver.set_current(prior).is_ok().then_some(prior);
            return Err(ContextActivationError {
                source,
                restored_prior,
            });
        }
        Ok(Self {
            driver: Arc::clone(driver),
            prior: Some(prior),
        })
    }

    pub(crate) fn keep_current(mut self) {
        self.prior = None;
    }

    pub(crate) fn restore(mut self) -> Result<(), DriverError> {
        if let Some(prior) = self.prior.take() {
            self.driver.set_current(prior)
        } else {
            Ok(())
        }
    }

    pub(crate) fn restore_before_owner_release(
        mut self,
        owner: sys::CUcontext,
    ) -> Result<(), DriverError> {
        let Some(prior) = self.prior.take() else {
            return Ok(());
        };
        // Restoring the final owner's handle and then releasing the last primary-context retain would leave a
        // destroyed context current, so unbind it instead (CUDA may reveal an older stacked context).
        self.driver
            .set_current(current_before_owner_release(prior, owner))
    }
}

impl ContextActivationError {
    fn prepare_owner_release(&self, driver: &dyn CudaDriver, owner: sys::CUcontext) -> bool {
        let Some(prior) = self.restored_prior else {
            return false;
        };
        let safe_current = current_before_owner_release(prior, owner);
        safe_current == prior || driver.set_current(safe_current).is_ok()
    }
}

impl Drop for CurrentContextGuard {
    fn drop(&mut self) {
        if let Some(prior) = self.prior.take() {
            let _ = self.driver.set_current(prior);
        }
    }
}

pub(crate) struct PtxOwner {
    pub(crate) driver: Arc<dyn CudaDriver>,
    pub(crate) device: sys::CUdevice,
    pub(crate) ctx: sys::CUcontext,
    pub(crate) stream: sys::CUstream,
    pub(crate) modules: Mutex<Vec<sys::CUmodule>>,
    /// Live/peak bytes and allocation count by role (Card 547a): the one memory service's view of
    /// every allocation this context has made.
    pub(crate) memory: poot_runtime_common::MemoryCounters,
}

// SAFETY: every field here is either a plain address (the `sys::CU*` handles and
// `MemoryCounters`, which is itself `Arc`-backed atomics) or behind its own internal
// synchronization (`modules: Mutex`). No method reads or writes a handle without first making
// `ctx` current on the calling thread (`current_guard`/`with_current`), so moving a `PtxOwner` to
// another thread (R471-010) is sound.
unsafe impl Send for PtxOwner {}
// SAFETY: NVIDIA's CUDA driver API is documented thread-safe for a context used from more than
// one thread (each thread calls `cuCtxSetCurrent` on itself before calling in), and `modules` is
// the only field here with Rust-level shared mutable state, behind a `Mutex`. `Sync` is required
// for `Arc<PtxOwner>` (hence every owner-holding handle: `PtxBuffer`, `PtxGraphExec`,
// `PtxContext`) to be `Send` without a second, broader `unsafe impl Send` above this module
// (R471-010's actual complaint), even though no code here ever shares a live `&PtxOwner` across
// threads concurrently in practice.
unsafe impl Sync for PtxOwner {}

impl PtxOwner {
    pub(crate) fn current_guard(&self) -> Result<CurrentContextGuard, DriverError> {
        CurrentContextGuard::activate(&self.driver, self.ctx).map_err(|error| error.source)
    }

    pub(crate) fn with_current<T>(
        &self,
        operation: impl FnOnce(&dyn CudaDriver) -> Result<T, DriverError>,
    ) -> Result<T, PtxError> {
        let guard = self.current_guard()?;
        let result = operation(self.driver.as_ref());
        guard.restore()?;
        result.map_err(PtxError::Driver)
    }
}

impl Drop for PtxOwner {
    fn drop(&mut self) {
        let Ok(guard) = CurrentContextGuard::activate(&self.driver, self.ctx) else {
            // Without a known successful activation, even identifying the context for destruction is unsafe. Leak
            // native state and the retain: destroying a possibly-current final owner is worse than a bounded leak.
            return;
        };
        for module in self
            .modules
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .drain(..)
            .rev()
        {
            self.driver.unload_module(module);
        }
        self.driver.destroy_stream(self.stream);
        if guard.restore_before_owner_release(self.ctx).is_ok() {
            self.driver.release_primary(self.device);
        }
    }
}

pub(crate) fn construct_owner(
    driver: Arc<dyn CudaDriver>,
    device: sys::CUdevice,
    retain: impl FnOnce() -> Result<sys::CUcontext, PtxError>,
    create_stream: impl FnOnce() -> Result<sys::CUstream, PtxError>,
) -> Result<Arc<PtxOwner>, PtxError> {
    let ctx = retain()?;
    let guard = match CurrentContextGuard::activate(&driver, ctx) {
        Ok(guard) => guard,
        Err(error) => {
            if error.prepare_owner_release(driver.as_ref(), ctx) {
                driver.release_primary(device);
            }
            return Err(PtxError::Driver(error.source));
        }
    };
    let stream = match create_stream() {
        Ok(stream) => stream,
        Err(error) => {
            if guard.restore_before_owner_release(ctx).is_ok() {
                driver.release_primary(device);
            }
            return Err(error);
        }
    };
    guard.keep_current();
    Ok(Arc::new(PtxOwner {
        driver,
        device,
        ctx,
        stream,
        modules: Mutex::new(Vec::new()),
        memory: poot_runtime_common::MemoryCounters::new(),
    }))
}
