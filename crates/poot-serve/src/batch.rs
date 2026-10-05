//! The scheduling pieces of the batched serving loop: slots, admit/evict decisions, KV-fit budgeting,
//! preemption and token commit. No engine loop runs on them at this commit, so nothing outside their
//! own tests calls them; the one scheduling loop over the driver builds on them.

pub(crate) mod commit;
pub(crate) mod kv_budget;
pub(crate) mod preempt;
pub(crate) mod schedule;
pub(crate) mod slot;
