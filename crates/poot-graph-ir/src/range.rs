//! Value ranges of I32 values: an interval analysis over the graph (POOT-726).
//!
//! Every I32 value gets one proven interval of the words it can hold. The facts have two sources:
//!
//! - the equations: `Iota(n)` is `[0, n)`, `ArgTopK` ids over an axis of extent `E` are `[0, E)`, a
//!   `BlockTopK` result is `[-1, blocks)`, a
//!   comparison is `[0, 2)`, `Select` the union of its two branches, `Add`/`Sub`/`Mul` interval
//!   arithmetic, a movement op keeps its operand's range, a `Gather` reads a subset of its table;
//! - the inputs: a slot declares its range at construction
//!   ([`crate::Builder::slot_ranged`]) and every binder checks the bound words against it, so a
//!   declared range is a checked fact, not a promise. A weight table declares its range the same way
//!   ([`crate::Builder::constant_ranged`]) and the weight binder checks every stored word at bind.
//!   Every other input is the whole `i32` range.
//!
//! The analysis proves what a graph computes, never what a caller claims: the in-graph index guard
//! ([`crate::Builder::guard_index_bounds`]) is the same primitives as any other code, and its output is
//! in `[0, upper)` through the `GeU` and `Select` rules alone. No shape is matched.
//!
//! I32 arithmetic is modulo 2^32, so an interval whose result leaves `i32` may have wrapped and the
//! analysis answers the whole `i32` range for it: a sound fact in both the wrapping and the faulting
//! reading of an overflow.
//!
//! `poot-graph-plan`'s `compile` reads this to refuse an `I32 -> F32` cast whose operand is not proven
//! inside the exactly representable integers of `f32` ([`IndexRange::F32_EXACT`]).

use std::collections::HashMap;

use crate::graph::{Graph, Operand, Storage, ValueId, ValueTable};
use crate::op::{BinOp, OpKind, RedOp, UnOp};
use crate::types::{DType, Scalar};

/// A half-open interval `[lo, hi_exclusive)` of `i32` words, with `lo < hi_exclusive`.
///
/// The bounds are `i64` so the exclusive upper bound of `i32::MAX` is representable and interval
/// arithmetic cannot overflow before it is compared against the `i32` range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IndexRange {
    lo: i64,
    hi_exclusive: i64,
}

impl IndexRange {
    /// Every `i32` word: what an I32 value is when nothing narrower is proven.
    pub const I32: IndexRange = IndexRange {
        lo: i32::MIN as i64,
        hi_exclusive: i32::MAX as i64 + 1,
    };

    /// The words `0` and `1`: an I32 flag (a real key against padding).
    pub const FLAG: IndexRange = IndexRange {
        lo: 0,
        hi_exclusive: 2,
    };

    /// The integers an `f32` represents exactly, `[-2^24, 2^24]`: the range a `Cast(I32 -> F32)`
    /// operand must be proven inside.
    pub const F32_EXACT: IndexRange = IndexRange {
        lo: -(1 << 24),
        hi_exclusive: (1 << 24) + 1,
    };

    /// `[lo, hi_exclusive)`, or `None` when it is empty or leaves the `i32` words.
    pub fn new(lo: i64, hi_exclusive: i64) -> Option<Self> {
        (lo < hi_exclusive && Self::I32.lo <= lo && hi_exclusive <= Self::I32.hi_exclusive)
            .then_some(IndexRange { lo, hi_exclusive })
    }

    /// `[0, upper_exclusive)`: the ids of an extent. `None` for an empty extent or one past `i32`.
    pub fn below(upper_exclusive: usize) -> Option<Self> {
        Self::new(0, i64::try_from(upper_exclusive).ok()?)
    }

    /// The smallest word of the range.
    pub fn lo(&self) -> i64 {
        self.lo
    }

    /// One past the largest word of the range.
    pub fn hi_exclusive(&self) -> i64 {
        self.hi_exclusive
    }

    /// Whether `word` is in the range.
    pub fn contains(&self, word: i32) -> bool {
        (self.lo..self.hi_exclusive).contains(&i64::from(word))
    }

    /// Whether every word of `self` is in `outer`.
    pub fn within(&self, outer: &IndexRange) -> bool {
        outer.lo <= self.lo && self.hi_exclusive <= outer.hi_exclusive
    }

    /// The smallest range holding both.
    pub fn union(&self, other: &IndexRange) -> IndexRange {
        IndexRange {
            lo: self.lo.min(other.lo),
            hi_exclusive: self.hi_exclusive.max(other.hi_exclusive),
        }
    }

    /// The words in both, `None` when none are.
    fn intersect(&self, other: &IndexRange) -> Option<IndexRange> {
        IndexRange::new(
            self.lo.max(other.lo),
            self.hi_exclusive.min(other.hi_exclusive),
        )
    }

    fn point(word: i32) -> IndexRange {
        let word = i64::from(word);
        IndexRange {
            lo: word,
            hi_exclusive: word + 1,
        }
    }

    /// `[lo, hi]` of the words, as the closed bounds interval arithmetic works on.
    fn closed(&self) -> (i64, i64) {
        (self.lo, self.hi_exclusive - 1)
    }

    /// The range of closed bounds `[lo, hi]`, widened to every `i32` word when they leave `i32`: the
    /// result may have wrapped.
    fn from_closed(lo: i64, hi: i64) -> IndexRange {
        Self::new(lo, hi + 1).unwrap_or(Self::I32)
    }

    fn nonnegative(&self) -> bool {
        self.lo >= 0
    }
}

impl std::fmt::Display for IndexRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {})", self.lo, self.hi_exclusive)
    }
}

/// What [`value_ranges`] proves about a graph: the range of every I32 value, and for each value the
/// guard conditions its range depends on.
///
/// A range has two kinds of source. A declared slot range is a bound-checked fact: the binder refuses a
/// word outside it, so nothing downstream can see one. A range that exists only because a `Select` took a
/// condition into account (an index guard's clamp) holds because an out-of-range lane was *replaced*; the
/// consumer then sees a valid-looking value where the input was bad. [`RangeFacts::clamps`] names those
/// conditions, so a consumer that must not run on a silently clamped lane can require that each one is
/// observed (the guard's validation witness).
#[derive(Clone, Debug)]
pub struct RangeFacts {
    ranges: ValueTable<Option<IndexRange>>,
    clamps: ValueTable<Vec<ValueId>>,
}

impl RangeFacts {
    /// The conditions of the `Select`s whose narrowing the range of `value` rests on, transitively:
    /// every guard that may have replaced a lane of `value`. Empty when the range comes from definitions
    /// and bound-checked slots alone.
    pub fn clamps(&self, value: ValueId) -> &[ValueId] {
        self.clamps.get(value).map_or(&[], Vec::as_slice)
    }
}

impl std::ops::Index<ValueId> for RangeFacts {
    type Output = Option<IndexRange>;

    fn index(&self, value: ValueId) -> &Option<IndexRange> {
        &self.ranges[value]
    }
}

/// The proven range of every I32 value of `graph` (`None` for a value of another dtype), with the guard
/// conditions each rests on.
///
/// One forward pass: equations are in definition order, so every operand's range is known when its
/// consumer is read.
pub fn value_ranges(graph: &Graph) -> RangeFacts {
    let mut ranges: ValueTable<Option<IndexRange>> = ValueTable::filled(graph.values().len(), None);
    for (id, meta) in graph.values().iter_enumerated() {
        if meta.aval.dtype != DType::I32 {
            continue;
        }
        ranges[id] = Some(match meta.storage {
            Storage::Computed(computed) => match computed {
                crate::graph::ComputedConst::Iota { len, .. } => {
                    IndexRange::below(len.max(1)).unwrap_or(IndexRange::I32)
                }
                _ => IndexRange::I32,
            },
            _ => meta.range().unwrap_or(IndexRange::I32),
        });
    }
    // The equation that defines each value: a `Select` reads its condition's comparison.
    let mut defs: ValueTable<Option<usize>> = ValueTable::filled(graph.values().len(), None);
    for (index, eqn) in graph.eqns().iter().enumerate() {
        defs[eqn.out] = Some(index);
    }
    let mut clamps: ValueTable<Vec<ValueId>> = ValueTable::filled(graph.values().len(), Vec::new());
    for eqn in graph.eqns() {
        if graph.aval(eqn.out).dtype != DType::I32 {
            continue;
        }
        let (range, narrowed_by) = match eqn.op {
            OpKind::Select => select_range(graph, &defs, &ranges, &eqn.inputs),
            _ => (
                equation_range(graph, &defs, &ranges, &eqn.op, &eqn.inputs),
                None,
            ),
        };
        ranges[eqn.out] = Some(range);
        let mut rests_on: Vec<ValueId> = eqn
            .inputs
            .iter()
            .filter_map(|operand| match operand {
                Operand::Value(id) => Some(clamps[*id].iter().copied()),
                Operand::Lit(_) => None,
            })
            .flatten()
            .chain(narrowed_by)
            .collect();
        rests_on.sort_unstable();
        rests_on.dedup();
        clamps[eqn.out] = rests_on;
    }
    RangeFacts { ranges, clamps }
}

/// The range of one operand: a literal is a point, a value its proven range. A non-I32 value (the
/// source of a `Cast`) has no word range of its own and reads as every `i32` word.
fn operand_range(ranges: &ValueTable<Option<IndexRange>>, operand: &Operand) -> IndexRange {
    match operand {
        Operand::Lit(Scalar::I32(word)) => IndexRange::point(*word),
        Operand::Lit(_) => IndexRange::I32,
        Operand::Value(id) => ranges[*id].unwrap_or(IndexRange::I32),
    }
}

/// The most any F32 bound may be for [`f32_bounds`] to report it: every whole number up to here is exact in
/// F32, so interval arithmetic in `f64` on such bounds is the F32 arithmetic.
const EXACT_F32: f64 = 16_777_216.0;

/// Closed bounds `[lo, hi]` on the value of an F32 operand, proven from its definition alone: a literal is a
/// point, a comparison is `[0, 1]`, a movement op keeps its operand's bounds, and a sum, difference or product
/// of bounded operands is the interval arithmetic of theirs. `None` for anything else, and for bounds that are
/// not whole numbers within [`EXACT_F32`] (rounding could then differ from the arithmetic here).
fn f32_bounds(
    graph: &Graph,
    defs: &ValueTable<Option<usize>>,
    operand: &Operand,
    memo: &mut HashMap<ValueId, Option<(f64, f64)>>,
) -> Option<(f64, f64)> {
    let id = match operand {
        Operand::Lit(Scalar::F32(x)) => return exact_bounds(f64::from(*x), f64::from(*x)),
        Operand::Lit(Scalar::I32(_)) => return None,
        Operand::Value(id) => *id,
    };
    if let Some(known) = memo.get(&id) {
        return *known;
    }
    let bounds = (|| {
        let eqn = &graph.eqns()[defs[id]?];
        let mut input = |index: usize| f32_bounds(graph, defs, eqn.inputs.get(index)?, memo);
        match eqn.op {
            OpKind::Binary(BinOp::Ge) => Some((0.0, 1.0)),
            OpKind::Binary(op @ (BinOp::Add | BinOp::Sub | BinOp::Mul)) => {
                let ((a_lo, a_hi), (b_lo, b_hi)) = (input(0)?, input(1)?);
                let corners = match op {
                    BinOp::Add => [a_lo + b_lo, a_hi + b_hi, a_lo + b_lo, a_hi + b_hi],
                    BinOp::Sub => [a_lo - b_hi, a_hi - b_lo, a_lo - b_hi, a_hi - b_lo],
                    _ => [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi],
                };
                exact_bounds(
                    corners.into_iter().fold(f64::INFINITY, f64::min),
                    corners.into_iter().fold(f64::NEG_INFINITY, f64::max),
                )
            }
            OpKind::Reshape { .. }
            | OpKind::Transpose { .. }
            | OpKind::Slice { .. }
            | OpKind::Broadcast { .. } => input(0),
            _ => None,
        }
    })();
    memo.insert(id, bounds);
    bounds
}

/// `[lo, hi]` when both are whole numbers within [`EXACT_F32`].
fn exact_bounds(lo: f64, hi: f64) -> Option<(f64, f64)> {
    let exact = |x: f64| x.fract() == 0.0 && x.abs() <= EXACT_F32;
    (exact(lo) && exact(hi)).then_some((lo, hi))
}

/// The range an I32-producing equation proves for its output.
fn equation_range(
    graph: &Graph,
    defs: &ValueTable<Option<usize>>,
    ranges: &ValueTable<Option<IndexRange>>,
    op: &OpKind,
    inputs: &[Operand],
) -> IndexRange {
    let at = |index: usize| {
        inputs
            .get(index)
            .map_or(IndexRange::I32, |operand| operand_range(ranges, operand))
    };
    let source_dtype = |index: usize| match inputs.get(index) {
        Some(Operand::Value(id)) => Some(graph.aval(*id).dtype),
        _ => None,
    };
    match op {
        OpKind::Iota { len, .. } => IndexRange::below((*len).max(1)).unwrap_or(IndexRange::I32),
        OpKind::ArgTopK { .. } => match inputs.first() {
            Some(Operand::Value(id)) => graph
                .aval(*id)
                .shape
                .last()
                .and_then(|&extent| IndexRange::below(extent))
                .unwrap_or(IndexRange::I32),
            _ => IndexRange::I32,
        },
        // The packed ids and indicator: ids `-1..blocks`, the indicator `0` or `1`.
        OpKind::BlockTopK(_) => inputs
            .first()
            .and_then(|operand| match operand {
                Operand::Value(id) => graph.aval(*id).shape.last().copied(),
                Operand::Lit(_) => None,
            })
            .and_then(|blocks| IndexRange::new(-1, i64::try_from(blocks.max(2)).ok()?))
            .unwrap_or(IndexRange::I32),
        OpKind::SampleToken { .. } => match inputs.first() {
            // `(token, non_finite_index)`: the token is a logits column; the second word is `-1` or the lowest
            // non-finite index, also a column.
            Some(Operand::Value(id)) => graph
                .aval(*id)
                .shape
                .last()
                .and_then(|&extent| IndexRange::new(-1, i64::try_from(extent).ok()?))
                .unwrap_or(IndexRange::I32),
            _ => IndexRange::I32,
        },
        // `x - x` is zero in every word, whatever `x` holds: the fallback of an index guard.
        OpKind::Binary(BinOp::Sub) if matches!(inputs, [Operand::Value(a), Operand::Value(b)] if a == b) => {
            IndexRange::point(0)
        }
        OpKind::Binary(op) => binary_range(*op, at(0), at(1)),
        OpKind::Unary(UnOp::Not) => {
            // `!x == -x - 1`.
            let (lo, hi) = at(0).closed();
            IndexRange::from_closed(-hi - 1, -lo - 1)
        }
        OpKind::Unary(UnOp::Clz) => IndexRange {
            lo: 0,
            hi_exclusive: 33,
        },
        OpKind::Cast { to: DType::I32 } => match source_dtype(0) {
            Some(DType::I32) => at(0),
            Some(DType::I8) => IndexRange {
                lo: -128,
                hi_exclusive: 128,
            },
            // An F32 built from comparisons and small literals (an indicator, its complement, a product of
            // them) holds small whole numbers, so its cast does too.
            Some(DType::F32) => match inputs.first() {
                Some(operand) => f32_bounds(graph, defs, operand, &mut HashMap::new())
                    .map_or(IndexRange::I32, |(lo, hi)| {
                        IndexRange::from_closed(lo as i64, hi as i64)
                    }),
                None => IndexRange::I32,
            },
            _ => IndexRange::I32,
        },
        // Movement and selection ops produce words their operand already holds.
        OpKind::Reshape { .. }
        | OpKind::Transpose { .. }
        | OpKind::Slice { .. }
        | OpKind::Broadcast { .. }
        | OpKind::Gather { .. }
        | OpKind::Reduce { op: RedOp::Max, .. }
        | OpKind::AllGather { .. }
        | OpKind::AllReduce { op: RedOp::Max, .. } => at(0),
        OpKind::Concat { .. } => inputs
            .iter()
            .map(|operand| operand_range(ranges, operand))
            .reduce(|left, right| left.union(&right))
            .unwrap_or(IndexRange::I32),
        // `operand` and `update` each supply words of the result.
        OpKind::DynamicUpdateSlice { .. } => at(0).union(&at(1)),
        OpKind::Scatter { .. } => at(0),
        OpKind::ScatterUpdate => at(0).union(&at(1)),
        OpKind::Reduce {
            op: RedOp::Sum,
            axis,
            ..
        } => match inputs.first() {
            Some(Operand::Value(id)) => {
                let extent = graph.aval(*id).shape.get(*axis).copied().unwrap_or(1);
                match i64::try_from(extent) {
                    Ok(extent) => {
                        let (lo, hi) = at(0).closed();
                        // The sum of `extent` words, each in `[lo, hi]`.
                        IndexRange::from_closed(
                            lo.saturating_mul(extent),
                            hi.saturating_mul(extent),
                        )
                    }
                    Err(_) => IndexRange::I32,
                }
            }
            _ => IndexRange::I32,
        },
        _ => IndexRange::I32,
    }
}

/// `Select(condition, if_true, if_false)`: the union of its branches, each narrowed by what the
/// condition implies in that branch about the value it compares.
///
/// A branch that is the very value the condition compares against an I32 literal is read under that
/// comparison: where `GeU(x, k)` is zero, `x` is below `k` as an unsigned word, so in `[0, k)` (one
/// unsigned test covers both bounds); where `Ge(x, k)` is zero, `x < k`, and where it is nonzero,
/// `x >= k`. This is how `Select(GeU(x, k), Sub(x, x), x)` is proven `[0, k)` without the shape being
/// matched as a whole: each equation of it is an ordinary rule.
fn select_range(
    graph: &Graph,
    defs: &ValueTable<Option<usize>>,
    ranges: &ValueTable<Option<IndexRange>>,
    inputs: &[Operand],
) -> (IndexRange, Option<ValueId>) {
    let [condition, if_true, if_false] = inputs else {
        return (IndexRange::I32, None);
    };
    let comparison = match condition {
        Operand::Value(id) => defs[*id].map(|index| &graph.eqns()[index]).and_then(|eqn| {
            match (&eqn.op, eqn.inputs.as_slice()) {
                (
                    OpKind::Binary(op @ (BinOp::Ge | BinOp::GeU)),
                    [Operand::Value(compared), Operand::Lit(Scalar::I32(bound))],
                ) => Some((*op, *compared, i64::from(*bound))),
                _ => None,
            }
        }),
        Operand::Lit(_) => None,
    };
    let narrowed = |branch: &Operand, condition_holds: bool| {
        let range = operand_range(ranges, branch);
        let Some((op, compared, bound)) = comparison else {
            return range;
        };
        if !matches!(branch, Operand::Value(id) if *id == compared) {
            return range;
        }
        let implied = match (op, condition_holds) {
            // `x <u k` for a positive bound is `0 <= x < k`.
            (BinOp::GeU, false) => IndexRange::new(0, bound),
            // `x >=u k` leaves a nonnegative `x` at or above `k`.
            (BinOp::GeU, true) if range.nonnegative() => IndexRange::new(bound, range.hi_exclusive),
            (BinOp::Ge, false) => IndexRange::new(range.lo, bound),
            (BinOp::Ge, true) => IndexRange::new(bound, range.hi_exclusive),
            _ => None,
        };
        implied
            .and_then(|implied| range.intersect(&implied))
            .unwrap_or(range)
    };
    let (taken, fallback) = (narrowed(if_true, true), narrowed(if_false, false));
    // The condition is a clamp when it tightened a branch: a lane of that branch outside the narrowed
    // range is replaced by the other branch rather than passed through.
    let tightened =
        taken != operand_range(ranges, if_true) || fallback != operand_range(ranges, if_false);
    let condition = match condition {
        Operand::Value(id) if tightened => Some(*id),
        _ => None,
    };
    (taken.union(&fallback), condition)
}

fn binary_range(op: BinOp, left: IndexRange, right: IndexRange) -> IndexRange {
    let (llo, lhi) = left.closed();
    let (rlo, rhi) = right.closed();
    match op {
        BinOp::Add => IndexRange::from_closed(llo + rlo, lhi + rhi),
        BinOp::Sub => IndexRange::from_closed(llo - rhi, lhi - rlo),
        BinOp::Mul => {
            // Bounds are `i32`-sized, so the four products fit `i64`.
            let products = [llo * rlo, llo * rhi, lhi * rlo, lhi * rhi];
            let lo = products.into_iter().min().unwrap_or(0);
            let hi = products.into_iter().max().unwrap_or(0);
            IndexRange::from_closed(lo, hi)
        }
        BinOp::Max => IndexRange::from_closed(llo.max(rlo), lhi.max(rhi)),
        BinOp::Ge | BinOp::GeU => IndexRange {
            lo: 0,
            hi_exclusive: 2,
        },
        // `x % d` for an unsigned divisor `d` is below `d`; a divisor range that is a positive `i32`
        // range is its own unsigned range.
        BinOp::RemU if right.nonnegative() => IndexRange::from_closed(0, (rhi - 1).max(0)),
        // `x & m` of a nonnegative mask is in `[0, m]`.
        BinOp::And => match (left.nonnegative(), right.nonnegative()) {
            (true, true) => IndexRange::from_closed(0, lhi.min(rhi)),
            (true, false) => IndexRange::from_closed(0, lhi),
            (false, true) => IndexRange::from_closed(0, rhi),
            (false, false) => IndexRange::I32,
        },
        // A logical right shift of a nonnegative word never grows it.
        BinOp::Shr if left.nonnegative() => IndexRange::from_closed(0, lhi),
        BinOp::Div | BinOp::RemU | BinOp::Or | BinOp::Xor | BinOp::Shl | BinOp::Shr => {
            IndexRange::I32
        }
    }
}

#[cfg(test)]
mod tests;
