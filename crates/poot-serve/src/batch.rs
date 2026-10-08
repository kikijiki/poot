//! The batched serving loop: one scheduling loop over the driver's `step` (`engine`), the slot a
//! request occupies (`slot`), the token commit (`commit`) and the per-step prefill budget (`schedule`).

pub(crate) mod commit;
pub(crate) mod engine;
pub(crate) mod schedule;
pub(crate) mod slot;
