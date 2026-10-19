//! The driver's prepared-set ledger: the entries it has compiled, the shared owners those entries
//! retain, the bytes they are charged and what preparing them taught the driver (the carried state's
//! residency, the published caps and the encode shapes admitted).
//!
//! An entry is retained until its adapter generation unloads; a shared owner (a constant several entries
//! reference) is charged once however many hold it. Nothing is recorded until [`PreparedLedger::hold`],
//! so a refusal or a failed load between a reservation and its hold leaves no trace.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;

use super::caps::{DriverCaps, StateResidency};
use super::encode::EncodeShapes;
use super::{EntryKey, Prepared, Retention};

/// What one prepared entry holds: its own bytes, and the keys of the shared owners it references.
#[derive(Clone, Debug)]
pub(super) struct Held {
    pub(super) private: u64,
    pub(super) owners: Vec<String>,
}

/// A retention the driver has measured but not yet recorded.
pub(super) struct Reservation {
    pub(super) held: Held,
    /// The bytes of each of `held.owners`, in order.
    pub(super) bytes: Vec<u64>,
    /// The bytes recording it adds: its own, plus each owner no entry holds yet.
    pub(super) charge: u64,
}

/// A shared owner and how many entries reference it.
#[derive(Clone, Copy, Debug)]
struct Owner {
    bytes: u64,
    holders: usize,
}

/// The prepared set and everything the driver learned by preparing it.
#[derive(Default)]
pub(super) struct PreparedLedger {
    pub(super) entries: HashMap<EntryKey, Prepared>,
    /// The row-gather entries beam search reorders its state with, by row count ([`super::beam`]).
    pub(super) reorders: HashMap<NonZeroUsize, Prepared>,
    /// The shared owners the prepared entries reference, by key.
    owners: HashMap<String, Owner>,
    /// The bytes held: every shared owner once, plus each entry's own.
    pub(super) retained_bytes: u64,
    /// Whether any prepared entry carries recurrent state; read from the traced graphs' state roles.
    pub(super) residency: StateResidency,
    /// The caps published by the last `prepare`; `None` until one admitted a shape.
    pub(super) caps: Option<DriverCaps>,
    /// The token-row shapes `prepare_encode` admitted; `None` for a driver that serves none.
    pub(super) encode_shapes: Option<EncodeShapes>,
}

impl PreparedLedger {
    /// Entries held, beam reorders included.
    pub(super) fn len(&self) -> usize {
        self.entries.len() + self.reorders.len()
    }

    /// What admitting `retention` would hold and add: every shared owner not yet held, once, plus the
    /// entry's own bytes.
    pub(super) fn reserve(&self, retention: Retention) -> Reservation {
        let mut seen = HashSet::new();
        let mut charge = retention.private;
        let mut owners = Vec::with_capacity(retention.shared.len());
        let mut bytes = Vec::with_capacity(retention.shared.len());
        for owner in retention.shared {
            if !seen.insert(owner.key.clone()) {
                continue;
            }
            if !self.owners.contains_key(&owner.key) {
                charge = charge.saturating_add(owner.bytes);
            }
            owners.push(owner.key);
            bytes.push(owner.bytes);
        }
        Reservation {
            held: Held {
                private: retention.private,
                owners,
            },
            bytes,
            charge,
        }
    }

    /// Record a reservation once its entry exists: take a hold on each shared owner and add the charge.
    pub(super) fn hold(&mut self, reservation: &Reservation) {
        for (key, &bytes) in reservation.held.owners.iter().zip(&reservation.bytes) {
            self.owners
                .entry(key.clone())
                .and_modify(|owner| owner.holders += 1)
                .or_insert(Owner { bytes, holders: 1 });
        }
        self.retained_bytes = self.retained_bytes.saturating_add(reservation.charge);
    }

    /// Drop an entry's holds: its own bytes, and each shared owner it was the last holder of.
    pub(super) fn drop_holds(&mut self, held: &Held) {
        let mut freed = held.private;
        for key in &held.owners {
            let Some(owner) = self.owners.get_mut(key) else {
                continue;
            };
            owner.holders -= 1;
            if owner.holders == 0 {
                freed = freed.saturating_add(owner.bytes);
                self.owners.remove(key);
            }
        }
        self.retained_bytes = self.retained_bytes.saturating_sub(freed);
    }
}
