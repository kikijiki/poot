//! The LoRA registry (Card 735, POOT-ADR-0112): the immutable adapter-set generations the driver serves
//! and the per-row adapter index each request binds.
//!
//! A *set* is what one prepared entry is compiled for: logical adapter names and factor shapes, plus the
//! immutable identity of the source that produced the weights. Two sets with the same names and shapes
//! but different sources are different generations, so a reloaded adapter never reuses an entry (or a
//! cached prefix block) built for the old weights. Inside one generation a request picks its adapter by
//! [`LoraIdx`], a bound input that never specializes an entry. The registry owns identity and lifetime
//! only; the graph transform that inserts the adapters is POOT-579's, applied once during whole-program
//! preparation.

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroU64};

use super::prefix_cache::PrefixIdentity;

/// The immutable identity of the files (or the loader incarnation) an adapter set was read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AdapterSource(pub NonZeroU64);

/// One adapter of a set: its logical name and the factor shape its entry is compiled for.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AdapterSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

/// An adapter set as captured: what a generation is keyed on.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AdapterSet {
    pub adapters: Vec<AdapterSpec>,
    pub source: AdapterSource,
}

/// One admitted adapter-set generation. Driver-issued and never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GenerationId(NonZeroU32);

/// A request's adapter within a generation, 1-based: 0 is the base weights (`NO_ADAPTER`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoraIdx(NonZeroU32);

impl LoraIdx {
    pub fn new(index: u32) -> Option<Self> {
        NonZeroU32::new(index).map(Self)
    }

    pub fn get(self) -> u32 {
        self.0.get()
    }
}

/// Which weights a sequence runs under.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AdapterRef {
    #[default]
    Base,
    Adapter {
        generation: GenerationId,
        index: LoraIdx,
    },
}

impl AdapterRef {
    /// The generation this reference holds, if any.
    pub fn generation(self) -> Option<GenerationId> {
        match self {
            AdapterRef::Base => None,
            AdapterRef::Adapter { generation, .. } => Some(generation),
        }
    }

    /// The row value bound to `Slot::LoraIdx`: 0 for the base weights.
    pub(crate) fn row_value(self) -> f32 {
        match self {
            AdapterRef::Base => 0.0,
            AdapterRef::Adapter { index, .. } => index.get() as f32,
        }
    }
}

/// Why the registry refuses an adapter operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LoraError {
    #[error("adapter generation {0:?} is not loaded")]
    UnknownGeneration(GenerationId),
    #[error("generation {generation:?} holds {adapters} adapters; index {index} is out of range")]
    IndexOutOfRange {
        generation: GenerationId,
        index: u32,
        adapters: usize,
    },
    /// A live request holds the generation: unloading it would free weights a sequence still runs.
    #[error("generation {generation:?} is held by {live} live sequences")]
    InUse {
        generation: GenerationId,
        live: usize,
    },
    #[error("the registry has issued every generation id")]
    Exhausted,
}

#[derive(Debug)]
struct Generation {
    set: AdapterSet,
    /// Open sequences holding this generation.
    live: usize,
}

/// The loaded adapter-set generations.
#[derive(Debug, Default)]
pub(crate) struct LoraRegistry {
    generations: HashMap<GenerationId, Generation>,
    by_set: HashMap<AdapterSet, GenerationId>,
    issued: u32,
}

impl LoraRegistry {
    /// Admit `set`: the generation already holding an identical set (same names, shapes and source), or
    /// a new one. A set that differs in source only is a new generation.
    pub(crate) fn capture(&mut self, set: AdapterSet) -> Result<GenerationId, LoraError> {
        if let Some(&id) = self.by_set.get(&set) {
            return Ok(id);
        }
        let issued = self.issued.checked_add(1).ok_or(LoraError::Exhausted)?;
        self.issued = issued;
        let id = GenerationId(NonZeroU32::new(issued).expect("the counter starts at one"));
        self.by_set.insert(set.clone(), id);
        self.generations.insert(id, Generation { set, live: 0 });
        Ok(id)
    }

    pub(crate) fn contains(&self, generation: GenerationId) -> bool {
        self.generations.contains_key(&generation)
    }

    /// A sequence takes `adapter`: validates it and counts the hold. Returns the effective-weights
    /// identity the prefix cache keys that sequence's blocks on.
    pub(crate) fn retain(&mut self, adapter: AdapterRef) -> Result<PrefixIdentity, LoraError> {
        let AdapterRef::Adapter { generation, index } = adapter else {
            return Ok(PrefixIdentity::BASE);
        };
        let held = self
            .generations
            .get_mut(&generation)
            .ok_or(LoraError::UnknownGeneration(generation))?;
        if index.get() as usize > held.set.adapters.len() {
            return Err(LoraError::IndexOutOfRange {
                generation,
                index: index.get(),
                adapters: held.set.adapters.len(),
            });
        }
        held.live += 1;
        Ok(prefix_identity(generation, index.get()))
    }

    /// A sequence gives `adapter` back.
    pub(crate) fn release(&mut self, adapter: AdapterRef) {
        if let Some(generation) = adapter.generation() {
            let held = self
                .generations
                .get_mut(&generation)
                .expect("a retained generation stays loaded while held");
            held.live -= 1;
        }
    }

    /// `generation` can be dropped: it is loaded and no open sequence holds it.
    pub(crate) fn check_unloadable(&self, generation: GenerationId) -> Result<(), LoraError> {
        let held = self
            .generations
            .get(&generation)
            .ok_or(LoraError::UnknownGeneration(generation))?;
        if held.live > 0 {
            return Err(LoraError::InUse {
                generation,
                live: held.live,
            });
        }
        Ok(())
    }

    /// The prefix identities of every adapter of `generation`.
    pub(crate) fn identities_of(&self, generation: GenerationId) -> Vec<PrefixIdentity> {
        let adapters = self
            .generations
            .get(&generation)
            .map_or(0, |held| held.set.adapters.len());
        (1..=adapters as u32)
            .map(|index| prefix_identity(generation, index))
            .collect()
    }

    /// Drop `generation`. Refused while any open sequence holds it.
    pub(crate) fn unload(&mut self, generation: GenerationId) -> Result<(), LoraError> {
        self.check_unloadable(generation)?;
        let held = self.generations.remove(&generation).expect("checked above");
        self.by_set.remove(&held.set);
        Ok(())
    }
}

/// The effective-weights identity the prefix cache keys `index` of `generation` on.
fn prefix_identity(generation: GenerationId, index: u32) -> PrefixIdentity {
    PrefixIdentity::adapter(u64::from(generation.0.get()) << 32 | u64::from(index))
}
