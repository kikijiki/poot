//! Executor submission mode, independent of compilation (Card 546a: renamed from
//! `ProductionPlanMode`, Z4). Compilation produces the same plans for either mode; the engine
//! records once and replays under `Replay` (the default) and submits every step under `Eager`.
//! `Eager` survives this card: the bench prefill arm and the PTX/ROCm/
//! VLM/speculative paths stay on it until Cards 546b/548/549 move them onto the contract, which
//! admits `Submission::Replay` only and typed-refuses an `Eager` program.

/// How a compiled program's steps reach the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Submission {
    /// Every step is submitted separately.
    Eager,
    /// The step is recorded once and replayed on later steps (the default).
    Replay,
}
