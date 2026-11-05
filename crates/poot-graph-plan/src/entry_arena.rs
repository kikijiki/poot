//! The activation arena of one staged entry (POOT-1433): the buffers every stage of an entry computes
//! its values into, shared across stages by lifetime.
//!
//! Each stage's [`BufferPlan`] colors that stage's values onto its own slots. The stages of an entry run
//! one after another, so a slot that is dead once its stage has run (every slot but the ones
//! [`BufferPlan::slot_outlives_stage`] names) is free for the next stage: the entry needs, per storage
//! class, as many shared buffers as its most demanding stage has slots, each as large as the largest
//! slot that ever sits in it. Matching the slots of every stage largest-first onto the shared buffers by
//! rank gives the smallest such sum.
//!
//! A slot that outlives its stage (an export a later stage binds, the primary output, a state-commit
//! source, a validation witness) keeps a buffer of its own for the whole entry, sized exactly to its one
//! value: the engine hands such a buffer to a later stage, a chained entry or the host with no copy and
//! no per-call length.
//!
//! Ordering assumption (every backend): the executor replays the stages of an entry in order, each
//! replay ordered after the previous one with its writes visible to it, the same ordering that already
//! lets a stage read the state and the exports an earlier stage wrote. A later stage's write to a
//! shared buffer is therefore ordered after the earlier stage's last read of it.

use std::collections::HashMap;

use poot_target::BufferStorage;

use crate::{ArenaSlotId, BufferPlan};

/// The shared buffers of one entry and where each stage's slots live in them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EntryArena {
    buffers: Vec<(BufferStorage, usize)>,
    stage_buffers: Vec<Vec<usize>>,
}

impl EntryArena {
    /// The arena for stages that run in the order `stages` lists their buffer plans.
    pub fn plan<'a>(stages: impl IntoIterator<Item = &'a BufferPlan>) -> Self {
        let mut buffers: Vec<(BufferStorage, usize)> = Vec::new();
        // The shared buffers of each storage class, in rank order (largest first within a stage).
        let mut shared: HashMap<BufferStorage, Vec<usize>> = HashMap::new();
        let mut stage_buffers = Vec::new();
        for plan in stages {
            let mut placed = vec![usize::MAX; plan.slot_count()];
            // In order of each class's first slot, so the buffer indices do not depend on hash order.
            let mut by_class: Vec<(BufferStorage, Vec<(usize, ArenaSlotId)>)> = Vec::new();
            for (slot, storage, elems) in plan.slots() {
                if plan.slot_outlives_stage(slot) {
                    placed[slot.index()] = buffers.len();
                    buffers.push((storage, elems));
                } else {
                    match by_class.iter_mut().find(|(class, _)| *class == storage) {
                        Some((_, slots)) => slots.push((elems, slot)),
                        None => by_class.push((storage, vec![(elems, slot)])),
                    }
                }
            }
            for (storage, mut slots) in by_class {
                slots.sort_by_key(|&(elems, slot)| (std::cmp::Reverse(elems), slot.index()));
                let ranks = shared.entry(storage).or_default();
                for (rank, (elems, slot)) in slots.into_iter().enumerate() {
                    let buffer = match ranks.get(rank) {
                        Some(&buffer) => buffer,
                        None => {
                            buffers.push((storage, 0));
                            ranks.push(buffers.len() - 1);
                            buffers.len() - 1
                        }
                    };
                    buffers[buffer].1 = buffers[buffer].1.max(elems);
                    placed[slot.index()] = buffer;
                }
            }
            stage_buffers.push(placed);
        }
        Self {
            buffers,
            stage_buffers,
        }
    }

    /// Every shared buffer: its storage and native device-element count, in buffer-index order.
    pub fn buffers(&self) -> &[(BufferStorage, usize)] {
        &self.buffers
    }

    /// The shared buffer index of `slot` of the stage at position `stage` in the order the arena was
    /// planned over.
    #[cfg(any(test, feature = "test-support"))]
    pub fn buffer_of(&self, stage: usize, slot: ArenaSlotId) -> usize {
        self.stage_buffers[stage][slot.index()]
    }

    /// The stage's slot-to-buffer table, indexed by [`ArenaSlotId::index`].
    pub fn stage_buffers(&self, stage: usize) -> &[usize] {
        &self.stage_buffers[stage]
    }

    /// Bytes of every shared buffer: what the entry allocates for locally-computed values.
    pub fn bytes(&self) -> usize {
        self.buffers
            .iter()
            .map(|&(storage, elems)| elems * storage.element().byte_width())
            .sum()
    }
}
