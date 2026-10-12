//! The forward write map of an in-place row write ([`crate::OpKind::ScatterUpdate`]).
//!
//! A forward map names, for each of the step's `N` source rows, the pool slot the row is written to, or
//! [`ForwardSlotMap::PADDING`] for a row that writes nothing. The kernel owns one thread per source row, so
//! two rows naming one slot are a write race, and a slot outside the pool is an out-of-bounds store. Both
//! are refused here, in one pass over the `N` entries, so a map that reaches a bind site cannot hold either:
//! [`ForwardSlotMap::new`] is the only constructor.

use std::collections::HashMap;

/// A validated forward slot map: every entry is [`Self::PADDING`] or a distinct slot of the pool it was
/// checked against. The default is the empty map, which names no slot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ForwardSlotMap {
    slots: Vec<i32>,
}

/// Why a candidate forward slot map is not a [`ForwardSlotMap`]. Each variant names the offending slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SlotMapError {
    /// An entry is neither the padding sentinel nor a slot of the pool.
    #[error("slot map entry {index} names slot {slot}, outside the pool of {pool_slots} slots")]
    OutOfRange {
        /// Position of the entry in the map.
        index: usize,
        /// The slot the entry names.
        slot: i32,
        /// Size of the pool the map was checked against.
        pool_slots: usize,
    },
    /// Two entries name one slot: two source rows would write it.
    #[error("slot map entries {first} and {second} both name slot {slot}")]
    Duplicate {
        /// The slot named twice.
        slot: i32,
        /// Position of the first entry naming it.
        first: usize,
        /// Position of the second entry naming it.
        second: usize,
    },
}

impl ForwardSlotMap {
    /// The entry of a source row that writes nothing.
    pub const PADDING: i32 = -1;

    /// Check `slots` against a pool of `pool_slots` slots: each entry is [`Self::PADDING`] or in
    /// `0..pool_slots`, and no slot is named twice. `O(slots.len())`.
    pub fn new(slots: Vec<i32>, pool_slots: usize) -> Result<Self, SlotMapError> {
        let mut first_owner: HashMap<i32, usize> = HashMap::with_capacity(slots.len());
        for (index, &slot) in slots.iter().enumerate() {
            if slot == Self::PADDING {
                continue;
            }
            if !usize::try_from(slot).is_ok_and(|slot| slot < pool_slots) {
                return Err(SlotMapError::OutOfRange {
                    index,
                    slot,
                    pool_slots,
                });
            }
            if let Some(first) = first_owner.insert(slot, index) {
                return Err(SlotMapError::Duplicate {
                    slot,
                    first,
                    second: index,
                });
            }
        }
        Ok(Self { slots })
    }

    /// The slots, one per source row.
    pub fn as_slice(&self) -> &[i32] {
        &self.slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_and_distinct_in_range_slots_are_accepted() {
        let map = ForwardSlotMap::new(vec![3, -1, 0, 7], 8).unwrap();
        assert_eq!(map.as_slice(), &[3, -1, 0, 7]);
        assert!(
            ForwardSlotMap::new(vec![-1, -1], 8)
                .unwrap()
                .as_slice()
                .iter()
                .all(|&s| s == ForwardSlotMap::PADDING)
        );
    }

    #[test]
    fn a_slot_named_twice_is_refused_with_both_entries() {
        assert_eq!(
            ForwardSlotMap::new(vec![2, 5, -1, 2], 8),
            Err(SlotMapError::Duplicate {
                slot: 2,
                first: 0,
                second: 3,
            })
        );
    }

    #[test]
    fn a_slot_outside_the_pool_is_refused_naming_it() {
        for bad in [8, 9, i32::MAX, -2, i32::MIN] {
            assert_eq!(
                ForwardSlotMap::new(vec![0, bad], 8),
                Err(SlotMapError::OutOfRange {
                    index: 1,
                    slot: bad,
                    pool_slots: 8,
                }),
                "{bad}"
            );
        }
    }
}
