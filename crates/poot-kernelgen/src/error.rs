//! `KernelGenError`: a public generator's own precondition on its shape or count arguments, violated
//! (R484-010). Before this type existed, a generator asserted its precondition and panicked inside the
//! planner chain; now it returns this instead, and every caller propagates it (`poot-graph-plan` as a
//! planner refusal, `poot-codegen`'s tests as an ordinary `Result`). A generator with no real
//! precondition (nothing in its argument shape can make its body malformed) stays infallible: wrapping it
//! in an always-`Ok` `Result` would be a null error type, not a contract.
//!
//! This is distinct from Card 531a's `Body::verify`: that checks the emitted IR is well-formed (types,
//! scopes, buffer bindings) after a `Body` exists; this checks the generator's own argument contract
//! before it builds one (a block size a tile decomposition depends on, a rank two shapes must share, an
//! index that must fit inside another extent).

/// A public kernel generator's precondition, violated. `generator` names the function (its own name, not
/// a caller's), so a match on the error variant plus `generator` says exactly which contract broke.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum KernelGenError {
    /// `value` is not a multiple of `divisor` (a block, group or tile size the generator's unpack or
    /// tiling decomposition depends on).
    #[error("{generator}: {dim} = {value} is not a multiple of {divisor}")]
    NotDivisible {
        generator: &'static str,
        dim: String,
        value: usize,
        divisor: usize,
    },
    /// `value` must equal `expected` exactly (a fixed dimension the generator's control flow, not merely
    /// its unpack, depends on).
    #[error("{generator}: {dim} = {value}, must equal {expected}")]
    NotEqualTo {
        generator: &'static str,
        dim: String,
        value: usize,
        expected: usize,
    },
    /// `value` is below the generator's minimum (a nonzero extent, a minimum rank, a minimum count).
    #[error("{generator}: {what} = {value}, need at least {min}")]
    BelowMinimum {
        generator: &'static str,
        what: String,
        value: usize,
        min: usize,
    },
    /// `value` exceeds the generator's bound (an index or length that must fit inside another extent).
    #[error("{generator}: {what} = {value}, exceeds bound {bound}")]
    ExceedsBound {
        generator: &'static str,
        what: String,
        value: usize,
        bound: usize,
    },
    /// `axis` is not a valid axis of a shape of rank `rank`.
    #[error("{generator}: axis {axis} is out of range for rank {rank}")]
    AxisOutOfRange {
        generator: &'static str,
        axis: usize,
        rank: usize,
    },
    /// Two counts the generator requires to agree (ranks, leaf counts, input counts) do not.
    #[error("{generator}: {what}: expected {expected}, got {actual}")]
    CountMismatch {
        generator: &'static str,
        what: String,
        expected: usize,
        actual: usize,
    },
    /// Two shapes the generator requires to agree (e.g. a concat's or scatter's non-axis dims) do not.
    #[error("{generator}: {what}: {a:?} vs {b:?}")]
    ShapeMismatch {
        generator: &'static str,
        what: String,
        a: Vec<usize>,
        b: Vec<usize>,
    },
    /// A [`poot_quant::format::FormatDescriptor`] shape the packed decode emitter (`crate::packed`)
    /// cannot state: a field of more than two pieces, or a lowering the current staging does not
    /// admit yet (dquant.md D4; 542a covers block storage without sub-block factors).
    #[error("packed_kernel: {format:?}: {reason}")]
    UnsupportedDescriptor {
        format: poot_quant::format::WeightFormat,
        reason: &'static str,
    },
    /// The target cannot state this request at all (not a violated argument: the arguments are fine, no
    /// body exists for them on this backend). `request` names the generator, `reason` what is missing.
    /// The planner turns it into a typed refusal at claim time.
    #[error("{request}: unsupported on this target: {reason}")]
    Unsupported {
        request: &'static str,
        reason: &'static str,
    },
    /// The body this request needs would exceed the caller's [`crate::BodyLimits`]. `attempted` is the
    /// total the generator would have reached, `limit` the cap; `stage` says whether the request was
    /// refused from its checked size before any statement existed or the finished body was measured.
    #[error("{generator}: body needs {attempted} {resource} against a limit of {limit} ({stage})")]
    BodyLimit {
        generator: &'static str,
        resource: BodyResource,
        stage: BodyStage,
        attempted: usize,
        limit: usize,
    },
}

/// What a [`KernelGenError::BodyLimit`] counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BodyResource {
    /// Statements plus block terminators.
    #[error("instructions")]
    Instructions,
    /// Locals, parameters included.
    #[error("locals")]
    Locals,
}

/// When a [`KernelGenError::BodyLimit`] was raised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BodyStage {
    /// From the request's checked conservative size, before a generator built anything.
    #[error("refused before generation")]
    Sizing,
    /// From the finished body: a generator whose size the request cannot bound ahead of time.
    #[error("measured after generation")]
    Finished,
}

impl KernelGenError {
    /// The generator whose precondition this violates, for a caller that logs or matches without
    /// destructuring every variant.
    pub fn generator(&self) -> &'static str {
        match self {
            Self::NotDivisible { generator, .. }
            | Self::NotEqualTo { generator, .. }
            | Self::BelowMinimum { generator, .. }
            | Self::ExceedsBound { generator, .. }
            | Self::AxisOutOfRange { generator, .. }
            | Self::CountMismatch { generator, .. }
            | Self::ShapeMismatch { generator, .. } => generator,
            Self::UnsupportedDescriptor { .. } => "packed_kernel",
            Self::Unsupported { request, .. } => request,
            Self::BodyLimit { generator, .. } => generator,
        }
    }
}
