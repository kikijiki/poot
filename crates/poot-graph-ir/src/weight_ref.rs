//! The typed identity of a weight constant: which mapped weight a `Storage::Const` value holds, and
//! which part of it.
//!
//! A tracer declares a model weight through [`crate::Builder::weight`], which records its
//! [`WeightId`] on the value's metadata ([`crate::ValueMeta::weight`]). The packed-weight binder and
//! legalize read that reference, never the constant's name, so a renamed constant still binds and a
//! chunk or carrier is derived from the weight it belongs to. The constant's name is a label
//! derived from the reference by [`WeightRef::name`]: the one spelling the executor's name-keyed
//! store lookup reads.

use poot_quant::SourceRole;
use poot_weights::WeightId;

use crate::packed_source::PackedSourceName;

/// Which mapped weight a constant holds, and which part of it: the weight as declared, one source
/// component of its packed storage, or a row chunk of either that legalize split off.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightRef {
    id: WeightId,
    source: Option<SourceRole>,
    chunk: Option<usize>,
}

impl WeightRef {
    /// The weight `id` as a tracer declares it.
    pub const fn whole(id: WeightId) -> Self {
        Self {
            id,
            source: None,
            chunk: None,
        }
    }

    /// The `role` source component of `id`'s packed storage: a carrier constant the packed-weight
    /// binder stages.
    #[must_use]
    pub const fn with_source(self, role: SourceRole) -> Self {
        Self {
            source: Some(role),
            ..self
        }
    }

    /// Chunk `index` of this constant's leading axis, as legalize splits an oversized constant.
    #[must_use]
    pub const fn with_chunk(self, index: usize) -> Self {
        Self {
            chunk: Some(index),
            ..self
        }
    }

    /// The mapped weight this constant is part of.
    pub const fn id(&self) -> WeightId {
        self.id
    }

    /// The packed source role this constant carries, when it is a carrier.
    pub const fn source(&self) -> Option<SourceRole> {
        self.source
    }

    /// The chunk index, when this constant is a chunk legalize split off.
    pub const fn chunk(&self) -> Option<usize> {
        self.chunk
    }

    /// The graph constant name this reference is spelled as: [`WeightId::const_name`], then the
    /// packed source suffix of a carrier, then `.chunkN` of a chunk.
    pub fn name(&self) -> String {
        let mut name = self.id.const_name();
        if let Some(role) = self.source {
            name = PackedSourceName::new(&name, role).into();
        }
        if let Some(index) = self.chunk {
            name = format!("{name}.chunk{index}");
        }
        name
    }
}

#[cfg(test)]
mod tests {
    use poot_weights::{AttnRole, WeightRole};

    use super::*;

    #[test]
    fn a_reference_spells_its_weight_carrier_and_chunk() {
        let id = WeightId::layer(3, WeightRole::Attn(AttnRole::Q));
        let whole = WeightRef::whole(id);
        assert_eq!(whole.name(), "w.l3.attn.q");
        let carrier = whole.with_source(SourceRole::Blocks);
        assert_eq!(carrier.name(), "w.l3.attn.q.packed_blocks_source");
        assert_eq!(
            carrier.with_chunk(2).name(),
            "w.l3.attn.q.packed_blocks_source.chunk2"
        );
        assert_eq!(whole.with_chunk(0).name(), "w.l3.attn.q.chunk0");
        assert_eq!(
            (whole.id(), whole.source(), whole.chunk()),
            (id, None, None)
        );
    }
}
