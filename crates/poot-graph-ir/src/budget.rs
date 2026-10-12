//! The caller-supplied construction budget of one trace: [`TraceLimits`], the typed refusal
//! [`TraceBudgetExceeded`] and the [`Meter`] every [`crate::Builder`] (and every append plan staged
//! against it) charges before it allocates or appends.
//!
//! Prior art (retrieved 2026-10-11): wasmparser's `limits.rs` checks a declared count against a named
//! limit before reading the items and reports the limit; wasmtime's `ResourceLimiter` is consulted before
//! a growth happens and the embedder supplies the numbers; vLLM refuses an oversize request with a typed
//! error naming the parameter and the attempted value. llama.cpp's fixed graph arena is the contrary
//! example: `GGML_ASSERT(cgraph->n_nodes < cgraph->size)` aborts the process. This module follows the
//! first three: explicit finite numbers from the caller, the attempted total in the error, the check
//! before the allocation, never an `assert!`.
//!
//! What is bounded: the number of requested tokens (before the tracer body runs), the values and
//! equations a graph holds, the bytes of inline scalar literals, and the *requested* owned storage of
//! the construction paths (value and equation records, names, operand lists, inference scratch, plan
//! snapshots, staged fragments, final declarations). Not bounded: allocator overhead, memory a model
//! allocates outside the builder, RSS, compiler memory or device memory (compile expansion has its own
//! limits).

use std::cell::Cell;
use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};

/// The resource a [`TraceLimits`] field bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BudgetResource {
    /// New tokens a traced step carries.
    Tokens,
    /// Values (input binders and equation results) of one graph.
    Values,
    /// Equations of one graph.
    Equations,
    /// Bytes of inline scalar literals (`Operand::Lit`).
    LiteralBytes,
    /// Requested owned storage of the construction paths.
    OwnedBytes,
}

/// The construction path that requested storage: where a refusal happened, named so a test can assert
/// the reservation sits before the allocation it guards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocationSite {
    /// A step's token count against `max_tokens`, before the tracer body runs.
    TokenRequest,
    /// A named input binder's value record and its name.
    InputRecord,
    /// An equation's value record, operand list and attribute storage.
    EquationRecord,
    /// The clone of an equation's operand types that inference reads.
    InferenceOperands,
    /// The output type inference allocates (charged at its upper bound, trued up afterwards).
    InferenceResult,
    /// The type table an append plan snapshots from its builder.
    PlanSnapshot,
    /// A value staged in an append plan.
    StagedValue,
    /// A name staged in an append plan.
    StagedName,
    /// An equation (operands and attributes) staged in an append plan.
    StagedEquation,
    /// The operand-type clone staged inference reads.
    StagedInference,
    /// The tables a preflight keeps (defined flags, staged names) while it validates a fragment.
    PreflightTables,
    /// The carried-state pairs declared at finish.
    FinalState,
    /// The validation witnesses declared at finish.
    FinalValidation,
    /// The exports declared at finish.
    FinalExports,
    /// The graph a resumed builder takes over.
    ResumedGraph,
}

/// The total a refused request would have reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attempted {
    /// The total `used + requested` the request would have reached, in the resource's own unit.
    Total(u64),
    /// `used + requested` does not fit `u64`: reported as such, never saturated.
    Overflow,
}

impl fmt::Display for Attempted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attempted::Total(total) => write!(f, "{total}"),
            Attempted::Overflow => f.write_str("a total beyond u64"),
        }
    }
}

/// A request would exceed one of the caller's [`TraceLimits`]. Nothing was allocated or appended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "trace budget exceeded: {resource:?} limit {limit}, {used} used, {requested} requested \
     (attempted {attempted}) at {site:?}"
)]
pub struct TraceBudgetExceeded {
    /// The limit that refused the request.
    pub resource: BudgetResource,
    /// The configured cap for `resource`, in its own unit (tokens, values, equations or bytes).
    pub limit: u64,
    /// What the meter already held for `resource` when the request arrived.
    pub used: u64,
    /// The amount the refused request asked for on top of `used`.
    pub requested: u64,
    /// `used + requested`, or [`Attempted::Overflow`] when that sum does not fit `u64`.
    pub attempted: Attempted,
    /// The allocation site that made the request.
    pub site: AllocationSite,
}

/// The finite construction limits of one trace, supplied by the caller of the trace: there is no
/// default and no unlimited value. A production caller states the numbers its workload needs (the
/// finite values measured on the 659 ISL=570 qwen2 workload are recorded in POOT-733); a test spells
/// its own through the `test-support` constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceLimits {
    /// The most new tokens one traced step may carry.
    pub max_tokens: NonZeroUsize,
    /// The most values (input binders and equation results) one graph may hold.
    pub max_values: NonZeroUsize,
    /// The most equations one graph may hold.
    pub max_eqns: NonZeroUsize,
    /// The most bytes of inline scalar literals one graph may carry.
    pub max_literal_bytes: NonZeroU64,
    /// The most bytes of owned storage the construction paths may request at once.
    pub max_owned_bytes: NonZeroU64,
}

impl TraceLimits {
    /// Admit a step of `tokens` new tokens, or refuse it before any tracer body runs.
    pub fn admit_tokens(&self, tokens: NonZeroUsize) -> Result<(), TraceBudgetExceeded> {
        let requested = tokens.get() as u64;
        let limit = self.max_tokens.get() as u64;
        if requested > limit {
            return Err(TraceBudgetExceeded {
                resource: BudgetResource::Tokens,
                limit,
                used: 0,
                requested,
                attempted: Attempted::Total(requested),
                site: AllocationSite::TokenRequest,
            });
        }
        Ok(())
    }

    /// Large finite limits for a test fixture that is not about the limits: every field is a real
    /// bound (a tracer that loops forever still stops), none is the type's maximum.
    #[cfg(any(test, feature = "test-support"))]
    pub fn for_tests() -> Self {
        Self::for_tests_with_tokens(NonZeroUsize::MIN.saturating_add((1 << 16) - 1))
    }

    /// [`Self::for_tests`] with `max_tokens` set.
    #[cfg(any(test, feature = "test-support"))]
    pub fn for_tests_with_tokens(max_tokens: NonZeroUsize) -> Self {
        TraceLimits {
            max_tokens,
            max_values: NonZeroUsize::MIN.saturating_add((1 << 22) - 1),
            max_eqns: NonZeroUsize::MIN.saturating_add((1 << 22) - 1),
            max_literal_bytes: NonZeroU64::MIN.saturating_add((1 << 28) - 1),
            max_owned_bytes: NonZeroU64::MIN.saturating_add((1 << 34) - 1),
        }
    }
}

/// One simultaneous request or charge across the four metered resources.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Charge {
    pub values: u64,
    pub eqns: u64,
    pub literal: u64,
    pub owned: u64,
}

impl Charge {
    pub(crate) fn owned(owned: u64) -> Self {
        Charge {
            owned,
            ..Charge::default()
        }
    }

    /// `self + other`, or `None` when a field overflows.
    pub(crate) fn checked_add(self, other: Charge) -> Option<Charge> {
        Some(Charge {
            values: self.values.checked_add(other.values)?,
            eqns: self.eqns.checked_add(other.eqns)?,
            literal: self.literal.checked_add(other.literal)?,
            owned: self.owned.checked_add(other.owned)?,
        })
    }
}

/// The charges a [`Builder`](crate::Builder) holds right now: what the budget has been asked for and
/// not given back. A test reads it to record the exact pre-existing charge before a controlled
/// allocation.
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Charged {
    /// Values (input binders and equation results) currently charged.
    pub values: u64,
    /// Equations currently charged.
    pub equations: u64,
    /// Bytes of inline scalar literals currently charged.
    pub literal_bytes: u64,
    /// Bytes of owned construction storage currently charged.
    pub owned_bytes: u64,
}

/// One reservation the meter granted: the allocation-site record a test asserts against.
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reservation {
    /// The allocation site the reservation was made for.
    pub site: AllocationSite,
    /// Values reserved.
    pub values: u64,
    /// Equations reserved.
    pub equations: u64,
    /// Bytes of inline scalar literals reserved.
    pub literal_bytes: u64,
    /// Bytes of owned construction storage reserved.
    pub owned_bytes: u64,
}

/// One change of the ledger, in order: a reservation granted or a charge given back.
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerEvent {
    /// A reservation the meter granted.
    Reserved(Reservation),
    /// A charge given back to the meter, for example when a staged fragment is dropped.
    Released(Charged),
}

/// A reader of one trace's ledger that outlives the builder (a `finish` consumes it).
#[cfg(any(test, feature = "test-support"))]
#[derive(Clone, Debug)]
pub struct Ledger {
    pub(crate) meter: std::rc::Rc<Meter>,
}

#[cfg(any(test, feature = "test-support"))]
impl Ledger {
    /// What is charged now.
    pub fn charged(&self) -> Charged {
        self.meter.charged()
    }

    /// Every reservation and release so far, in order.
    pub fn events(&self) -> Vec<LedgerEvent> {
        self.meter.events()
    }
}

/// The shared ledger of one trace. A [`Builder`](crate::Builder) and every append plan staged against it
/// hold the same meter, so simultaneously live snapshots and fragments count together.
#[derive(Debug)]
pub(crate) struct Meter {
    limits: Cell<TraceLimits>,
    used: Cell<Charge>,
    #[cfg(any(test, feature = "test-support"))]
    log: std::cell::RefCell<Vec<LedgerEvent>>,
}

impl Meter {
    pub(crate) fn new(limits: TraceLimits) -> Self {
        Meter {
            limits: Cell::new(limits),
            used: Cell::new(Charge::default()),
            #[cfg(any(test, feature = "test-support"))]
            log: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// The refusal for a requested size that does not fit `u64`: the computation of the request itself
    /// overflowed, so the attempted total is [`Attempted::Overflow`].
    pub(crate) fn size_overflow(&self, site: AllocationSite) -> TraceBudgetExceeded {
        TraceBudgetExceeded {
            resource: BudgetResource::OwnedBytes,
            limit: self.limit_of(BudgetResource::OwnedBytes),
            used: self.used.get().owned,
            requested: u64::MAX,
            attempted: Attempted::Overflow,
            site,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn limits(&self) -> TraceLimits {
        self.limits.get()
    }

    fn limit_of(&self, resource: BudgetResource) -> u64 {
        match resource {
            BudgetResource::Tokens => self.limits.get().max_tokens.get() as u64,
            BudgetResource::Values => self.limits.get().max_values.get() as u64,
            BudgetResource::Equations => self.limits.get().max_eqns.get() as u64,
            BudgetResource::LiteralBytes => self.limits.get().max_literal_bytes.get(),
            BudgetResource::OwnedBytes => self.limits.get().max_owned_bytes.get(),
        }
    }

    /// Grant `want` or refuse it whole: no resource changes unless every one fits. The check is
    /// `used + requested` in checked `u64`; a sum that does not fit is [`Attempted::Overflow`].
    pub(crate) fn reserve(
        &self,
        want: Charge,
        site: AllocationSite,
    ) -> Result<(), TraceBudgetExceeded> {
        let used = self.used.get();
        for (resource, used, requested) in [
            (BudgetResource::Values, used.values, want.values),
            (BudgetResource::Equations, used.eqns, want.eqns),
            (BudgetResource::LiteralBytes, used.literal, want.literal),
            (BudgetResource::OwnedBytes, used.owned, want.owned),
        ] {
            let limit = self.limit_of(resource);
            let attempted = match used.checked_add(requested) {
                Some(total) if total <= limit => continue,
                Some(total) => Attempted::Total(total),
                None => Attempted::Overflow,
            };
            return Err(TraceBudgetExceeded {
                resource,
                limit,
                used,
                requested,
                attempted,
                site,
            });
        }
        // Every field passed `used + requested <= limit`, so the sum below cannot overflow.
        if let Some(next) = used.checked_add(want) {
            self.used.set(next);
        }
        #[cfg(any(test, feature = "test-support"))]
        self.log
            .borrow_mut()
            .push(LedgerEvent::Reserved(Reservation {
                site,
                values: want.values,
                equations: want.eqns,
                literal_bytes: want.literal,
                owned_bytes: want.owned,
            }));
        Ok(())
    }

    /// Give back a charge a transient owner (inference scratch, a dropped plan, a snapshot) held.
    /// Releasing more than was charged is a defect of the caller's bookkeeping; the ledger floors at
    /// zero rather than wrapping, and a debug build says so.
    pub(crate) fn release(&self, charge: Charge) {
        let used = self.used.get();
        debug_assert!(
            used.values >= charge.values
                && used.eqns >= charge.eqns
                && used.literal >= charge.literal
                && used.owned >= charge.owned,
            "released {charge:?} against a ledger holding {used:?}"
        );
        #[cfg(any(test, feature = "test-support"))]
        if charge != Charge::default() {
            self.log.borrow_mut().push(LedgerEvent::Released(Charged {
                values: charge.values,
                equations: charge.eqns,
                literal_bytes: charge.literal,
                owned_bytes: charge.owned,
            }));
        }
        self.used.set(Charge {
            values: used.values.saturating_sub(charge.values),
            eqns: used.eqns.saturating_sub(charge.eqns),
            literal: used.literal.saturating_sub(charge.literal),
            owned: used.owned.saturating_sub(charge.owned),
        });
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn charged(&self) -> Charged {
        let used = self.used.get();
        Charged {
            values: used.values,
            equations: used.eqns,
            literal_bytes: used.literal,
            owned_bytes: used.owned,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn events(&self) -> Vec<LedgerEvent> {
        self.log.borrow().clone()
    }
}

/// A charge held by a transient owner that gives it back when dropped, unless the owner's storage
/// became part of the graph first ([`HeldCharge::keep`]).
#[derive(Debug)]
pub(crate) struct HeldCharge {
    meter: std::rc::Rc<Meter>,
    charge: Charge,
}

impl HeldCharge {
    pub(crate) fn new(meter: std::rc::Rc<Meter>) -> Self {
        HeldCharge {
            meter,
            charge: Charge::default(),
        }
    }

    pub(crate) fn meter(&self) -> &Meter {
        &self.meter
    }

    /// Reserve `want` and hold it. On refusal nothing is held and the earlier holdings are untouched.
    pub(crate) fn reserve(
        &mut self,
        want: Charge,
        site: AllocationSite,
    ) -> Result<(), TraceBudgetExceeded> {
        self.meter.reserve(want, site)?;
        // `reserve` proved `used + want` fits, and `charge <= used`, so this cannot overflow.
        if let Some(total) = self.charge.checked_add(want) {
            self.charge = total;
        }
        Ok(())
    }

    /// Give back `part` now and stop holding it.
    pub(crate) fn release(&mut self, part: Charge) {
        let part = Charge {
            values: part.values.min(self.charge.values),
            eqns: part.eqns.min(self.charge.eqns),
            literal: part.literal.min(self.charge.literal),
            owned: part.owned.min(self.charge.owned),
        };
        self.meter.release(part);
        self.charge = Charge {
            values: self.charge.values - part.values,
            eqns: self.charge.eqns - part.eqns,
            literal: self.charge.literal - part.literal,
            owned: self.charge.owned - part.owned,
        };
    }

    /// The held storage became the graph's: stop holding it without giving it back.
    pub(crate) fn keep(&mut self) -> Charge {
        std::mem::take(&mut self.charge)
    }
}

impl Drop for HeldCharge {
    fn drop(&mut self) {
        self.meter.release(self.charge);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(values: usize, eqns: usize, literal: u64, owned: u64) -> TraceLimits {
        TraceLimits {
            max_tokens: NonZeroUsize::new(4).unwrap(),
            max_values: NonZeroUsize::new(values).unwrap(),
            max_eqns: NonZeroUsize::new(eqns).unwrap(),
            max_literal_bytes: NonZeroU64::new(literal).unwrap(),
            max_owned_bytes: NonZeroU64::new(owned).unwrap(),
        }
    }

    #[test]
    fn a_request_summing_past_u64_reports_overflow_not_a_saturated_total() {
        let meter = Meter::new(limits(8, 6, 32, u64::MAX));
        meter
            .reserve(Charge::owned(u64::MAX - 1), AllocationSite::StagedName)
            .expect("fits");
        let refused = meter
            .reserve(Charge::owned(2), AllocationSite::StagedName)
            .expect_err("u64::MAX - 1 + 2 does not fit u64");
        assert_eq!(refused.attempted, Attempted::Overflow);
        assert_eq!(refused.resource, BudgetResource::OwnedBytes);
        assert_eq!(refused.used, u64::MAX - 1);
        assert_eq!(refused.requested, 2);
    }

    #[test]
    fn a_refused_request_changes_no_resource() {
        let meter = Meter::new(limits(8, 6, 32, 1000));
        meter
            .reserve(Charge::owned(10), AllocationSite::InputRecord)
            .expect("fits");
        // Values fit, owned bytes do not: the value count must not move.
        let refused = meter
            .reserve(
                Charge {
                    values: 1,
                    owned: 991,
                    ..Charge::default()
                },
                AllocationSite::InputRecord,
            )
            .expect_err("1001 > 1000");
        assert_eq!(refused.resource, BudgetResource::OwnedBytes);
        assert_eq!(refused.attempted, Attempted::Total(1001));
        assert_eq!(meter.used(), Charge::owned(10));
    }

    #[test]
    fn tokens_admit_the_exact_boundary_and_refuse_the_next() {
        let limits = limits(8, 6, 32, 1000);
        assert!(limits.admit_tokens(NonZeroUsize::new(4).unwrap()).is_ok());
        let refused = limits
            .admit_tokens(NonZeroUsize::new(5).unwrap())
            .expect_err("5 > 4");
        assert_eq!(refused.resource, BudgetResource::Tokens);
        assert_eq!(refused.attempted, Attempted::Total(5));
    }

    #[test]
    fn a_held_charge_is_given_back_on_drop_unless_kept() {
        let meter = std::rc::Rc::new(Meter::new(limits(8, 6, 32, 1000)));
        {
            let mut held = HeldCharge::new(meter.clone());
            held.reserve(Charge::owned(100), AllocationSite::PlanSnapshot)
                .expect("fits");
            assert_eq!(meter.used(), Charge::owned(100));
        }
        assert_eq!(meter.used(), Charge::default());
        {
            let mut held = HeldCharge::new(meter.clone());
            held.reserve(Charge::owned(100), AllocationSite::StagedValue)
                .expect("fits");
            held.keep();
        }
        assert_eq!(meter.used(), Charge::owned(100));
    }

    impl Meter {
        pub(crate) fn set_limits(&self, limits: TraceLimits) {
            self.limits.set(limits);
        }

        pub(crate) fn used(&self) -> Charge {
            self.used.get()
        }
    }
}
