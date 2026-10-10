//! `PtxDevice` as a participant in a multi-device program: [`poot_executor::Collectives`] over the
//! runtime's peer copies and stream events.
//!
//! A [`poot_executor::Coordinator`] over several `PtxDevice`s records each stage into the stage's own
//! CUDA graph. The producing stage's graph ends in the peer copy of its exported value into the
//! consuming device's import buffer. Ordering between devices is not captured: the coordinator records
//! an event on the producing device's stream right after launching the stage's graph and makes the
//! consuming device's stream wait for it right before launching the consumer, both at the host call,
//! so a wait always names the current step's record. A graph retains the buffers it names until it is
//! destroyed; the coordinator owns the events.
//!
//! Only routes the hardware connects directly are admitted: a copy that would stage through the host
//! cannot be captured, so [`Collectives::open_route`] refuses it and no fallback exists.

use poot_executor::{Collectives, NativeCounters, SupportBound};
use poot_ptx_runtime::PtxSyncEvent;

use crate::PtxGpuError;
use crate::device::PtxDevice;

/// Conservative bound on the host and device memory one instantiated CUDA graph of a stage holds. The
/// driver does not report the size of a graph; a stage's graph is a few hundred nodes of launch
/// parameters, so one mebibyte covers it with a wide margin.
const GRAPH_BYTES_BOUND: u64 = 1 << 20;

/// Conservative bound on one CUDA event.
const EVENT_BYTES_BOUND: u64 = 4 << 10;

impl Collectives for PtxDevice {
    type Event = PtxSyncEvent;

    fn native_counters(&self) -> NativeCounters {
        let native = self.context().native_counters();
        NativeCounters {
            graph_instantiations: native.graph_instantiations,
            graph_launches: native.graph_launches,
            events: native.events,
            modules: native.modules,
            peer_copies: native.peer_copies,
        }
    }

    fn support_bound(&self) -> SupportBound {
        SupportBound {
            graph_bytes: GRAPH_BYTES_BOUND,
            event_bytes: EVENT_BYTES_BOUND,
        }
    }

    fn open_route(&mut self, peer: &Self) -> Result<(), PtxGpuError> {
        Ok(self.context().enable_peer_access(peer.context())?)
    }

    fn create_event(&mut self) -> Result<PtxSyncEvent, PtxGpuError> {
        Ok(self.context().create_sync_event()?)
    }

    fn record_event(&mut self, event: &PtxSyncEvent) -> Result<(), PtxGpuError> {
        Ok(self.context().record_sync_event(event)?)
    }

    fn wait_event(&mut self, event: &PtxSyncEvent) -> Result<(), PtxGpuError> {
        Ok(self.context().wait_sync_event(event)?)
    }

    fn peer_copy(
        &mut self,
        src: &poot_ptx_runtime::PtxBuffer,
        dst: &poot_ptx_runtime::PtxBuffer,
        bytes: usize,
    ) -> Result<(), PtxGpuError> {
        Ok(self.context().copy_to_peer(src, dst, bytes)?)
    }
}
