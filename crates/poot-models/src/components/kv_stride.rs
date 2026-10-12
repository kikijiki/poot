//! The stored extent of one KV head: the layout rule that keeps a cache's head stride off the DRAM channel
//! period (POOT-1150).
//!
//! A `[rows, kv_heads, capacity, head_dim]` f32 cache has the head stride `capacity * head_dim * 4`. When that
//! is a multiple of 2048 B (eight 256 B granules: [`poot_target::dram_channels::camps`]) the decode heads
//! reading the same position sit a whole channel period apart and camp on one or two of the APU's sixteen
//! channels (POOT-1139). A [`HeadSpan`] is the one place a cache's stored positions are decided: it is either
//! a stride that does not camp, or it names the [`Exemption`] that lets it camp. A cache state is declared
//! from a span, so an unaudited camping stride cannot be written down by accident, and the stride audit
//! reads the same type to know which camping rows are expected.
//!
//! The extra position of a padded span is never written and never visible: attention hides it with the mask.
//!
//! Why the rule is kept: the contiguous padding measured no decode gain on a real model (inside the +-1% noise
//! floor of the box, `receipts/1150-kv-stride`). It stays because the stride audit is a gate against the
//! known channel-camping layout: without the rule a model whose capacity or head dimension lands on a bad
//! stride would silently camp, and the rows the audit expects would no longer be exactly the exempt ones.
//! The cost is one extra position per head. The paged pool row stride is the one camping layout still
//! exempt ([`Exemption::PagedView`]).

use poot_target::dram_channels::camps;

/// Bytes of one cache element: the KV caches are f32.
const ELEMENT_BYTES: usize = size_of::<f32>();

/// Why a span may keep a camping stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exemption {
    /// One head row (`head_dim` elements) is itself a multiple of 2048 B, so no whole-position pad escapes
    /// the period.
    WholePeriodRow,
    /// The cache's consumers index it by logical position (a block-pooled index cache, a latent cache):
    /// they read exactly `capacity` positions, so the extent cannot grow.
    LogicalExtent,
    /// The cache is a view of a paged pool, read in place. Its stride is the pool's row stride
    /// (`kv_heads * head_dim * 4`), not a per-head capacity stride, so a position pad cannot move it.
    PagedView,
}

/// How many positions one head of a KV cache stores for a logical `capacity`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadSpan {
    capacity: usize,
    head_dim: usize,
    stored: usize,
    exemption: Option<Exemption>,
}

impl HeadSpan {
    /// The span of a `capacity`-position head of `head_dim` elements that attention itself reads: `capacity`
    /// when its stride reaches enough channels, else one position more. Only a head row that is itself a
    /// multiple of 2048 B cannot be padded off the period, and that span carries
    /// [`Exemption::WholePeriodRow`].
    pub fn padded(capacity: usize, head_dim: usize) -> Self {
        let stride = |positions: usize| positions * head_dim * ELEMENT_BYTES;
        let stored = if !camps(stride(capacity)) || camps(stride(capacity + 1)) {
            capacity
        } else {
            capacity + 1
        };
        Self::checked(capacity, head_dim, stored, Exemption::WholePeriodRow)
    }

    /// The span of a head stored at exactly `capacity` positions, for `reason`. The exemption is recorded only
    /// when that stride camps: an exact capacity that does not camp needs none.
    pub fn exact(capacity: usize, head_dim: usize, reason: Exemption) -> Self {
        Self::checked(capacity, head_dim, capacity, reason)
    }

    /// A span at `stored` positions: `reason` is kept only if the stride camps.
    fn checked(capacity: usize, head_dim: usize, stored: usize, reason: Exemption) -> Self {
        let camping = camps(stored * head_dim * ELEMENT_BYTES);
        Self {
            capacity,
            head_dim,
            stored,
            exemption: camping.then_some(reason),
        }
    }

    /// Positions the head stores; the key axis length of the cache.
    pub fn stored(self) -> usize {
        self.stored
    }

    /// Positions stored past the logical capacity: never written, always masked.
    pub fn pad(self) -> usize {
        self.stored - self.capacity
    }

    /// The exemption this span camps under, `None` when its stride does not camp.
    pub fn exemption(self) -> Option<Exemption> {
        self.exemption
    }

    /// Whether this span's head stride camps.
    pub fn camps(self) -> bool {
        camps(self.stored * self.head_dim * ELEMENT_BYTES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A power-of-two capacity at a power-of-two head width camps and gains one position; an off-period
    /// capacity keeps its exact one. Mutation: return `capacity` unconditionally from `padded`; the 2048 x 64
    /// and 2048 x 128 rows go red.
    #[test]
    fn a_camping_head_stride_gains_one_position() {
        let pad = |capacity, head_dim| HeadSpan::padded(capacity, head_dim).pad();
        // The serving shapes of POOT-1139: Qwen2.5-0.5B (head 64) and Qwen3-0.6B (head 128), 2048 positions.
        assert_eq!(pad(2048, 64), 1);
        assert_eq!(pad(2048, 128), 1);
        assert_eq!(pad(4096, 64), 1);
        // 2049 * 64 * 4 = 524_544 B: already off the period.
        assert_eq!(pad(2049, 64), 0);
        // 8 positions of 64 f32 are 2048 B; 7 are not.
        assert_eq!(pad(8, 64), 1);
        assert_eq!(pad(7, 64), 0);
        // A 512-element head row is 2048 B itself: no whole-position pad escapes it.
        assert_eq!(pad(2048, 512), 0);
    }

    /// A span camps only under an exemption, and a padded span only under the whole-period one. Mutation:
    /// return `capacity` from `padded` whenever it camps; the 64-wide rows camp under an exemption that
    /// is not `WholePeriodRow`'s (the head row is 256 B) and go red.
    #[test]
    fn a_span_never_camps_without_an_exemption() {
        for head_dim in [64, 80, 96, 128, 192, 256, 512, 1024] {
            for capacity in [1, 7, 8, 64, 100, 1024, 2048, 2049, 4096, 32768] {
                let padded = HeadSpan::padded(capacity, head_dim);
                assert_eq!(padded.camps(), padded.exemption().is_some());
                assert_eq!(
                    padded.camps(),
                    head_dim % 512 == 0,
                    "{capacity} x {head_dim}"
                );
                for reason in [Exemption::LogicalExtent, Exemption::PagedView] {
                    let exact = HeadSpan::exact(capacity, head_dim, reason);
                    assert_eq!(exact.pad(), 0);
                    assert_eq!(exact.camps(), exact.exemption() == Some(reason));
                }
            }
        }
    }
}
