use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

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

/// How many native resources one context has created and how many graph launches it has issued: the
/// numbers a warm replay must leave unchanged. A snapshot of [`OwnerCounters`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeCounters {
    /// CUDA graphs instantiated from a capture.
    pub graph_instantiations: u64,
    /// `cuGraphLaunch` calls.
    pub graph_launches: u64,
    /// CUDA events created (cross-device ordering events and device-timing pairs).
    pub events: u64,
    /// PTX modules loaded.
    pub modules: u64,
    /// Peer copies recorded into a capture.
    pub peer_copies: u64,
}

/// The live counters behind [`NativeCounters`]; `Relaxed` is enough because they only count.
#[derive(Default)]
pub(crate) struct OwnerCounters {
    pub(crate) graph_instantiations: AtomicU64,
    pub(crate) graph_launches: AtomicU64,
    pub(crate) events: AtomicU64,
    pub(crate) modules: AtomicU64,
    pub(crate) peer_copies: AtomicU64,
}

impl OwnerCounters {
    pub(crate) fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> NativeCounters {
        NativeCounters {
            graph_instantiations: self.graph_instantiations.load(Ordering::Relaxed),
            graph_launches: self.graph_launches.load(Ordering::Relaxed),
            events: self.events.load(Ordering::Relaxed),
            modules: self.modules.load(Ordering::Relaxed),
            peer_copies: self.peer_copies.load(Ordering::Relaxed),
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
    pub(crate) counters: OwnerCounters,
    /// The first device fault any call observed. Once set, the owner refuses every driver call.
    pub(crate) lost: OnceLock<DriverError>,
}

/// Whether a CUDA error means the device or context cannot run work any more. These are the
/// sticky errors: every later call on the context returns the same one.
fn is_device_fault(error: DriverError) -> bool {
    use sys::CUresult as R;
    matches!(
        error.0,
        R::CUDA_ERROR_ECC_UNCORRECTABLE
            | R::CUDA_ERROR_ILLEGAL_ADDRESS
            | R::CUDA_ERROR_LAUNCH_TIMEOUT
            | R::CUDA_ERROR_ASSERT
            | R::CUDA_ERROR_HARDWARE_STACK_ERROR
            | R::CUDA_ERROR_ILLEGAL_INSTRUCTION
            | R::CUDA_ERROR_MISALIGNED_ADDRESS
            | R::CUDA_ERROR_INVALID_ADDRESS_SPACE
            | R::CUDA_ERROR_INVALID_PC
            | R::CUDA_ERROR_LAUNCH_FAILED
            | R::CUDA_ERROR_NVLINK_UNCORRECTABLE
    )
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
    /// The error for a driver failure a call observed: a device fault poisons the owner and becomes
    /// [`PtxError::DeviceLost`], anything else stays a plain driver error.
    pub(crate) fn observe(&self, error: DriverError) -> PtxError {
        if is_device_fault(error) {
            let _ = self.lost.set(error);
            PtxError::DeviceLost(error)
        } else {
            PtxError::Driver(error)
        }
    }

    /// Refuse a call on a poisoned owner before it touches the driver.
    pub(crate) fn ensure_healthy(&self) -> Result<(), PtxError> {
        match self.lost.get() {
            Some(&cause) => Err(PtxError::Poisoned { cause }),
            None => Ok(()),
        }
    }

    pub(crate) fn current_guard(&self) -> Result<CurrentContextGuard, DriverError> {
        CurrentContextGuard::activate(&self.driver, self.ctx).map_err(|error| error.source)
    }

    pub(crate) fn with_current<T>(
        &self,
        operation: impl FnOnce(&dyn CudaDriver) -> Result<T, DriverError>,
    ) -> Result<T, PtxError> {
        self.ensure_healthy()?;
        let guard = self.current_guard()?;
        let result = operation(self.driver.as_ref());
        guard.restore()?;
        result.map_err(|error| self.observe(error))
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
        counters: OwnerCounters::default(),
        lost: OnceLock::new(),
    }))
}
