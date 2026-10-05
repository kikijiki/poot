//! Buffer-plan lifetimes and arena slots (Card 547b, amends R14/E6).
//!
//! `plan_buffers` gives every locally-computed value of a compiled graph a lifetime (the span of
//! equation indices it must stay valid for) and colors disjoint lifetimes onto the same [`ArenaSlotId`],
//! so the engine allocates one device buffer per slot instead of one per value (R471-005: no backend
//! plans buffer lifetimes until this card). A slot's buffer is sized to the largest value ever assigned
//! to it; a smaller value sharing it must never let a kernel see the slot's capacity as its own length
//! - that half of the fix is [`crate::Program::planned`]'s callers reading each
//! argument's own element count instead of the bound buffer's, never this module's concern.
//!
//! Two values never share a slot unless their planned [`poot_target::BufferStorage`] is identical: a
//! device buffer's storage is fixed at allocation, and binding one value's bytes through another's
//! `BufferStorage` is exactly the representation mismatch `poot_target::BufferStorage` exists to catch
//! (R484-001/R471-009) - so the arena coloring below runs independently per storage class.
//!
//! A value the engine commits with a whole-buffer `Device::copy` or reads back with a whole-buffer
//! `Device::read` (the primary output, a state-commit source, or a validation witness) needs its
//! buffer's capacity to equal its own logical size exactly: those two primitives have no per-call
//! length, unlike a kernel dispatch's [`crate::Plan`]-carried argument. Such a value always gets a
//! freshly allocated, never-reused slot (see `is_export` below), so nothing it was never sized for can
//! ever widen it.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use poot_graph_ir::{Graph, ValidationChannel, ValueId};
use poot_target::BufferStorage;

use crate::{GraphStorage, Plan, StateCommitPlan, ValidationPacketPlan, value_ids};

/// One arena slot: a device buffer reused by every locally-computed value assigned to it (never two
/// at once - their lifetimes are disjoint by construction), sized to the largest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArenaSlotId(u32);

impl ArenaSlotId {
    fn new(index: usize) -> Self {
        Self(index as u32)
    }

    /// This slot's index into [`BufferPlan::slot_storage`]/[`BufferPlan::slot_elems`], for a caller
    /// that holds one device buffer per slot in a plain `Vec` (the engine's own arena).
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug)]
struct SlotInfo {
    storage: BufferStorage,
    /// The largest occupant's native device-element count (post lane-encoding, e.g. two BF16 logical
    /// elements per packed `u32`: the same unit [`crate::Program::storage`] consumers already pass to
    /// `Device::allocate`), never a logical tensor `numel` this slot's smaller occupants may fall short
    /// of.
    elems: usize,
}

/// Lifetimes and arena-slot assignment for one compiled graph's locally-computed values (Card 547b).
/// Built once by `plan_buffers` and carried on [`crate::Program`]; a graph input (state, const, or
/// slot), a value donated in place into state, and a value whose lane this crate does not device-encode
/// (E4M3FN/RawBytes/I8) have no slot - the engine resolves the first two without asking here and
/// refuses to load the third, exactly as it did before arena slots existed.
#[derive(Clone, Debug, Default)]
pub struct BufferPlan {
    slots: Vec<SlotInfo>,
    assignment: HashMap<ValueId, ArenaSlotId>,
}

impl BufferPlan {
    /// Every arena slot this program needs, in slot-index order.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Every arena slot this program needs, with the one storage and native device-element count the
    /// engine allocates its one buffer with (Card 547b): the id this iterator hands out is the only
    /// way to construct an [`ArenaSlotId`] outside [`BufferPlan::slot`] itself, so a caller can never
    /// name a slot this plan never assigned.
    pub fn slots(&self) -> impl Iterator<Item = (ArenaSlotId, BufferStorage, usize)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .map(|(index, info)| (ArenaSlotId::new(index), info.storage, info.elems))
    }

    /// The storage every occupant of `slot` shares (arena coloring never mixes two distinct
    /// [`BufferStorage`]s onto one slot).
    pub fn slot_storage(&self, slot: ArenaSlotId) -> BufferStorage {
        self.slots[slot.index()].storage
    }

    /// `slot`'s native device-element count: the largest its occupants ever needed, what the engine
    /// passes to `Device::allocate` for this slot's one buffer.
    pub fn slot_elems(&self, slot: ArenaSlotId) -> usize {
        self.slots[slot.index()].elems
    }

    /// `slot`'s byte size (SC-001/SC-003): `slot_elems` native elements at `slot_storage`'s own
    /// element width.
    pub fn slot_bytes(&self, slot: ArenaSlotId) -> usize {
        self.slots[slot.index()].byte_size()
    }

    /// The arena slot backing `value`'s own buffer, or `None` when `value` has no local allocation at
    /// all (a graph input, a state-donated output, or an E4M3/raw-byte lane the engine already refuses
    /// to load). An `Alias`/`View` output is never a key here: it shares its root's `Loc` directly, so
    /// a caller that already resolves aliases before asking (as `crate::compile`'s one production
    /// consumer, `poot_executor::Engine::load_entry`, does) never needs this module to resolve them
    /// again.
    pub fn slot(&self, value: ValueId) -> Option<ArenaSlotId> {
        self.assignment.get(&value).copied()
    }

    /// Sum of every arena slot's bytes (SC-001, SC-003, SC-004): this program's peak locally-computed-
    /// value memory, distinct from the weight, state, and slot-input buffers the engine allocates
    /// separately and never arena-shares.
    pub fn arena_bytes(&self) -> usize {
        self.slots.iter().map(SlotInfo::byte_size).sum()
    }
}

impl SlotInfo {
    fn byte_size(&self) -> usize {
        self.elems * self.storage.element().byte_width()
    }
}

/// Lifetime and arena-slot assignment for every locally-computed value of `g` under `plans` (Card
/// 547b). `storage` is the program's own per-value storage ([`crate::value_storage_of_plans`]'s
/// result); `state_commit` and `validation` are the program's already-derived plans, read only to find
/// the values this program's epilogue exports by a whole-buffer `copy`/`read` (see the module doc).
///
/// A value's lifetime spans the equation index that materializes it (`Plan::Compute`/`ComputeMeta`/
/// `ComputeChunks`/`Collective`; `Alias`/`View` never materialize - a read of them extends their
/// root's lifetime instead, the same alias resolution [`crate::state_commit::state_commit_from_plans`]
/// already runs) to the last equation that reads it, inclusive: an equation that both reads a value
/// last and produces a new one is one dispatch with distinct input/output buffers, so the two are
/// always treated as live at once and never share a slot.
pub(crate) fn plan_buffers<V: ValidationChannel>(
    g: &Graph<V>,
    plans: &[Plan],
    storage: &GraphStorage,
    state_commit: &StateCommitPlan,
    validation: &ValidationPacketPlan,
) -> BufferPlan {
    let n = g.eqns.len();
    let mut roots: Vec<ValueId> = (0..g.values.len()).collect();
    let mut birth: HashMap<ValueId, usize> = HashMap::new();
    let mut death: HashMap<ValueId, usize> = HashMap::new();
    let mut elems: HashMap<ValueId, usize> = HashMap::new();
    let mut store_of: HashMap<ValueId, BufferStorage> = HashMap::new();

    for (eqn_idx, (eqn, plan)) in g.eqns.iter().zip(plans).enumerate() {
        match plan {
            Plan::Alias(source) => {
                roots[eqn.out] = roots[*source];
                continue;
            }
            Plan::View { src, .. } => {
                roots[eqn.out] = roots[*src];
                continue;
            }
            _ => {}
        }
        roots[eqn.out] = eqn.out;
        // Donated in place: `Engine::load_entry`'s `out_loc` writes this output
        // straight into the paired state buffer, never a local allocation - no slot to assign.
        if state_commit.inplace.contains_key(&eqn.out) {
            continue;
        }
        let value_storage = storage.storage(eqn.out).buffer_storage();
        let Some(value_elems) = value_storage.device_elems(g.aval(eqn.out).numel()) else {
            // E4M3/raw-byte lanes: `Engine::load_entry`'s own `allocate` refuses this value the same
            // way at load time (`LoadError::Unimplemented`), so leaving it unassigned here changes
            // nothing a caller could observe.
            continue;
        };
        birth.insert(eqn.out, eqn_idx);
        death.insert(eqn.out, eqn_idx);
        elems.insert(eqn.out, value_elems);
        store_of.insert(eqn.out, value_storage);
    }

    // Every operand read extends its root's death to the reading equation's index (inputs feeding the
    // eqn that also reads them keep `death == birth`, i.e. alive for only their own dispatch).
    for (eqn_idx, eqn) in g.eqns.iter().enumerate() {
        for value in value_ids(eqn) {
            let root = roots[value];
            if let Some(d) = death.get_mut(&root) {
                *d = (*d).max(eqn_idx);
            }
        }
    }

    // Exports (SC-004's "logical extent" guarantee, module doc): alive past the dispatch loop, into
    // the epilogue's whole-buffer `copy`/`read` - forced past every eqn index so nothing reuses the
    // slot afterward, and (`is_export`, below) never handed a slot anything else ever occupied first.
    let mut is_export: HashSet<ValueId> = HashSet::new();
    let mut export = |value: ValueId| {
        let root = roots[value];
        if let Some(d) = death.get_mut(&root) {
            *d = n;
            is_export.insert(root);
        }
    };
    export(g.output);
    for copy in &state_commit.copies {
        export(copy.state_output);
    }
    for source in &validation.sources {
        export(source.value);
    }

    // Linear-scan arena coloring (Card 547b), independently per `BufferStorage` class (a slot never
    // mixes two - module doc), birth order within a class, ties by `ValueId` for a deterministic plan.
    let mut groups: HashMap<BufferStorage, Vec<ValueId>> = HashMap::new();
    for &value in birth.keys() {
        groups.entry(store_of[&value]).or_default().push(value);
    }
    let mut group_keys: Vec<BufferStorage> = groups.keys().copied().collect();
    group_keys.sort_by_key(|key| {
        groups[key]
            .iter()
            .map(|&v| (birth[&v], v))
            .min()
            .expect("every group has at least the member that created it")
    });

    let mut slots: Vec<SlotInfo> = Vec::new();
    let mut assignment: HashMap<ValueId, ArenaSlotId> = HashMap::new();

    for key in group_keys {
        let mut members = groups.remove(&key).expect("key came from this map");
        members.sort_by_key(|&v| (birth[&v], v));

        let mut free: Vec<ArenaSlotId> = Vec::new();
        let mut active: BinaryHeap<Reverse<(usize, u32)>> = BinaryHeap::new();

        for value in members {
            let b = birth[&value];
            while let Some(&Reverse((d, slot_index))) = active.peek() {
                if d < b {
                    active.pop();
                    free.push(ArenaSlotId::new(slot_index as usize));
                } else {
                    break;
                }
            }
            let needed = elems[&value];
            let slot = if is_export.contains(&value) {
                None
            } else {
                free.pop()
            };
            let slot = match slot {
                Some(slot) => {
                    let info = &mut slots[slot.index()];
                    info.elems = info.elems.max(needed);
                    slot
                }
                None => {
                    let slot = ArenaSlotId::new(slots.len());
                    slots.push(SlotInfo {
                        storage: key,
                        elems: needed,
                    });
                    slot
                }
            };
            active.push(Reverse((death[&value], slot.0)));
            assignment.insert(value, slot);
        }
    }

    BufferPlan { slots, assignment }
}
