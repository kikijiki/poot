use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::sys;

use super::buffer::PtxBuffer;
use super::error::PtxError;
use super::owner::PtxOwner;

/// An instantiated, replayable CUDA graph (the captured dispatch sequence). [`PtxGraphExec::launch`] re-issues
/// the recorded sequence with one host call (`cuGraphLaunch`, G3c). Replays use the device addresses recorded
/// at capture, so refresh per-token scalar slots in place (via [`PtxContext::update_f32`]) before each launch.
///
/// The graph owns a handle to every buffer a captured dispatch referenced, so no replay can reach freed device
/// memory: dropping the caller's handles leaves the allocations alive until the graph drops.
///
/// [`PtxContext::update_f32`]: crate::PtxContext::update_f32
pub struct PtxGraphExec {
    pub(crate) exec: sys::CUgraphExec,
    pub(crate) graph: sys::CUgraph,
    pub(crate) owner: Arc<PtxOwner>,
    /// Released after `Drop` destroys the graph that records their addresses.
    #[expect(
        dead_code,
        reason = "owned so every captured buffer outlives the graph that records its address"
    )]
    captured: Vec<PtxBuffer>,
}

// SAFETY: `owner` is `Send + Sync` (see its doc) and `captured` is `Vec<PtxBuffer>`, `Send`
// automatically now that `PtxBuffer` is `Arc`-backed. `exec`/`graph` are plain graph addresses;
// every method that touches them (`launch`, `synchronize`, `Drop`) scopes `owner`'s context
// current on the calling thread first, so moving a `PtxGraphExec` to another thread (R471-010) is
// sound the same way moving a `PtxContext` is (see the crate doc).
unsafe impl Send for PtxGraphExec {}

impl PtxGraphExec {
    /// Replay the captured graph on its stream with a single host call. The originating context is current only
    /// for this call; the calling thread's prior state is restored. Does not sync.
    pub fn launch(&self) -> Result<(), PtxError> {
        self.owner
            .with_current(|driver| driver.launch_graph(self.exec, self.owner.stream))
    }

    /// Wait for this graph's originating stream. Usable even when the graph outlives its `PtxContext` handle
    /// (the graph retains the stream and primary context). Like [`Self::launch`], it scopes the originating
    /// context and restores the prior thread state.
    pub fn synchronize(&self) -> Result<(), PtxError> {
        self.owner
            .with_current(|driver| driver.synchronize_stream(self.owner.stream))
    }
}

impl Drop for PtxGraphExec {
    fn drop(&mut self) {
        if let Ok(_guard) = self.owner.current_guard() {
            if !self.exec.is_null() {
                self.owner.driver.destroy_graph_exec(self.exec);
            }
            if !self.graph.is_null() {
                self.owner.driver.destroy_graph(self.graph);
            }
        }
    }
}

pub(crate) struct GraphConstruction {
    pub(crate) graph: sys::CUgraph,
    pub(crate) exec: sys::CUgraphExec,
    pub(crate) owner: Arc<PtxOwner>,
    captured: Vec<PtxBuffer>,
}

impl GraphConstruction {
    pub(crate) fn finish(mut self) -> PtxGraphExec {
        let graph = std::mem::replace(&mut self.graph, std::ptr::null_mut());
        let exec = std::mem::replace(&mut self.exec, std::ptr::null_mut());
        PtxGraphExec {
            exec,
            graph,
            owner: Arc::clone(&self.owner),
            captured: std::mem::take(&mut self.captured),
        }
    }
}

impl Drop for GraphConstruction {
    fn drop(&mut self) {
        if self.exec.is_null() && self.graph.is_null() {
            return;
        }
        if let Ok(_guard) = self.owner.current_guard() {
            if !self.exec.is_null() {
                self.owner.driver.destroy_graph_exec(self.exec);
            }
            if !self.graph.is_null() {
                self.owner.driver.destroy_graph(self.graph);
            }
        }
    }
}

/// Instantiate `graph`, handing it the buffers its capture referenced (see [`CaptureRetention`]).
pub(crate) fn instantiate_graph(
    owner: Arc<PtxOwner>,
    graph: sys::CUgraph,
    captured: Vec<PtxBuffer>,
    instantiate: impl FnOnce(&mut sys::CUgraphExec) -> Result<(), PtxError>,
) -> Result<PtxGraphExec, PtxError> {
    let mut construction = GraphConstruction {
        graph,
        exec: std::ptr::null_mut(),
        owner,
        captured,
    };
    instantiate(&mut construction.exec)?;
    Ok(construction.finish())
}

/// The buffers an open stream capture has recorded device addresses of. A captured graph replays those raw
/// addresses, so every dispatch recorded during the capture retains its buffers here, and `end_capture` moves
/// them into the [`PtxGraphExec`]. `None` means no capture is open.
#[derive(Default)]
pub(crate) struct CaptureRetention(RefCell<Option<Vec<PtxBuffer>>>);

impl CaptureRetention {
    /// Start recording; called once the driver has opened the capture.
    pub(crate) fn begin(&self) {
        *self.0.borrow_mut() = Some(Vec::new());
    }

    pub(crate) fn is_open(&self) -> bool {
        self.0.borrow().is_some()
    }

    /// Keep `buffer` alive for the graph being captured. A no-op when no capture is open.
    pub(crate) fn retain(&self, buffer: &PtxBuffer) {
        if let Some(retained) = self.0.borrow_mut().as_mut() {
            retained.push(buffer.clone());
        }
    }

    /// Stop recording and return what the capture retained.
    pub(crate) fn finish(&self) -> Vec<PtxBuffer> {
        self.0.borrow_mut().take().unwrap_or_default()
    }
}

pub(crate) struct ModuleConstruction {
    module: sys::CUmodule,
    owner: Arc<PtxOwner>,
}

impl ModuleConstruction {
    pub(crate) fn new(module: sys::CUmodule, owner: Arc<PtxOwner>) -> Self {
        Self { module, owner }
    }

    pub(crate) fn finish(mut self) {
        let module = std::mem::replace(&mut self.module, std::ptr::null_mut());
        self.owner
            .modules
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(module);
    }
}

impl Drop for ModuleConstruction {
    fn drop(&mut self) {
        if !self.module.is_null()
            && let Ok(_guard) = self.owner.current_guard()
        {
            self.owner.driver.unload_module(self.module);
        }
    }
}

pub(crate) struct PtxEvent {
    pub(crate) event: sys::CUevent,
    pub(crate) owner: Arc<PtxOwner>,
}

impl PtxEvent {
    pub(crate) fn new(event: sys::CUevent, owner: Arc<PtxOwner>) -> Self {
        Self { event, owner }
    }
}

impl Drop for PtxEvent {
    fn drop(&mut self) {
        if !self.event.is_null()
            && let Ok(_guard) = self.owner.current_guard()
        {
            self.owner.driver.destroy_event(self.event);
        }
    }
}

/// A device-timed span around one [`PtxGraphExec`] launch (Card 552 SC-006), returned by
/// [`crate::PtxContext::launch_timed`]: two owned CUDA events, read by
/// [`crate::PtxContext::span_elapsed`]. Opaque to callers outside this crate; it exists only to be
/// handed back to `span_elapsed`.
pub struct PtxSpan {
    pub(crate) start: PtxEvent,
    pub(crate) stop: PtxEvent,
}

pub(crate) fn create_event_pair(
    owner: Arc<PtxOwner>,
    mut create: impl FnMut() -> Result<sys::CUevent, PtxError>,
) -> Result<(PtxEvent, PtxEvent), PtxError> {
    let start = PtxEvent::new(create()?, Arc::clone(&owner));
    let stop = PtxEvent::new(create()?, owner);
    Ok((start, stop))
}
