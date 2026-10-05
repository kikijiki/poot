//! The replay contract (FR-006): the static buffer, dispatch, and communication shape a later captured
//! executor must satisfy. The model-free fake runtime used to check cold planning versus warm replay is
//! test-owned under `multi_device::tests`.

use crate::multi_device::communication::CommunicationPlan;

/// A stable identity for one static buffer this plan places on a device or in a communication path (e.g.
/// `"weight:device0"`, `"comm:allreduce-0"`). Card 360 does not care what the string means, only that the
/// same buffer keeps the same id across cold plan and warm replay.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StaticBufferId(pub String);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticBufferKind {
    PackedWeight,
    Scale,
    DenseConstant,
    /// A resident carrier structure (e.g. a packed-row carrier) - rebuilding one is accounted separately
    /// from an ordinary reupload, per FR-006's own wording ("carrier rebuild").
    Carrier,
    Communication,
}

/// One static buffer this plan's replay contract must track for reuse across replays.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticBuffer {
    pub id: StaticBufferId,
    pub kind: StaticBufferKind,
    pub bytes: u64,
    /// A content fingerprint: the fake runtime treats an unchanged fingerprint as warm, a changed one as
    /// needing reupload/rebuild.
    pub fingerprint: u64,
    /// ADR-0099 item 3: the caller rewrites this buffer's content before every replay (token id,
    /// position, sequence length). Every other buffer keeps capture-fixed content; only an input may
    /// be written between replays, and only an input may change the bytes the captured graph reads.
    pub per_replay_input: bool,
}

/// The static buffer/dispatch/communication shape a later captured executor must satisfy. Only runtime
/// SLOT values (not modeled here - this crate is model-free) may vary between replays; everything in this
/// contract is fixed once planned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayContract {
    pub buffers: Vec<StaticBuffer>,
    /// The number of Card 356 dispatches this plan issues per replay (a fake-runtime count, not a real
    /// backend call).
    pub dispatch_count: usize,
    pub communication: CommunicationPlan,
}
