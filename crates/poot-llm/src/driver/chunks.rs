//! The exact prefill chunk plan (Card 734).
//!
//! Alignment is semantic, not a size cap: every prefill piece starts at and is sized in multiples of the
//! model's `prefill_granule`, and a tail below one granule runs as decode steps. The plan emits full
//! `prefill_chunk` pieces, then decomposes the remaining whole granules into descending powers of two,
//! so for one immutable scope (and one capacity, row count, layout and head) the prefill token sizes the
//! driver ever asks for are the full chunk plus the power-of-two multiples of the granule not above it:
//! a finite, enumerable set ([`admitted_pieces`]). The chunk is also bounded by the caller's
//! `max_tokens` independently of alignment: an aligned chunk above the limit is refused.

use std::num::NonZeroUsize;

use crate::driver::error::InvalidOptions;

/// One step of a prompt's plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Piece {
    /// A prefill step over this many prompt tokens.
    Prefill(NonZeroUsize),
    /// One prompt token below the granule, run as a decode step.
    Decode,
}

/// The validated chunking of a prompt: `chunk` is a multiple of `granule` and not above `max_tokens`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChunkPolicy {
    chunk: NonZeroUsize,
    granule: NonZeroUsize,
}

impl ChunkPolicy {
    pub(crate) fn new(
        chunk: NonZeroUsize,
        granule: NonZeroUsize,
        max_tokens: NonZeroUsize,
    ) -> Result<Self, InvalidOptions> {
        if !chunk.get().is_multiple_of(granule.get()) {
            return Err(InvalidOptions::ChunkNotGranuleAligned {
                chunk: chunk.get(),
                granule: granule.get(),
            });
        }
        if chunk > max_tokens {
            return Err(InvalidOptions::ChunkAboveTraceLimit {
                chunk: chunk.get(),
                max_tokens: max_tokens.get(),
            });
        }
        Ok(Self { chunk, granule })
    }

    /// The pieces covering a prompt of `prompt_len` tokens, in execution order.
    pub(crate) fn plan(&self, prompt_len: usize) -> Vec<Piece> {
        let chunk = self.chunk.get();
        let granule = self.granule.get();
        let mut pieces = vec![Piece::Prefill(self.chunk); prompt_len / chunk];
        let tail = prompt_len % chunk;
        let mut granules = tail / granule;
        let mut power = granules.checked_ilog2().map(|bits| 1usize << bits);
        while let Some(p) = power {
            if granules >= p {
                pieces.push(Piece::Prefill(
                    NonZeroUsize::new(p * granule).expect("a power of two times a granule"),
                ));
                granules -= p;
            }
            power = (p > 1).then_some(p / 2);
        }
        pieces.extend(std::iter::repeat_n(Piece::Decode, tail % granule));
        pieces
    }

    /// Every piece any prompt can plan to, each once: the full chunk, the power-of-two multiples of the
    /// granule below it, and the decode step when the granule is above one.
    pub(crate) fn admitted_pieces(&self) -> Vec<Piece> {
        let chunk = self.chunk.get();
        let granule = self.granule.get();
        let mut pieces = vec![Piece::Prefill(self.chunk)];
        let mut power = 1usize;
        while power * granule < chunk {
            pieces.push(Piece::Prefill(
                NonZeroUsize::new(power * granule).expect("a power of two times a granule"),
            ));
            power *= 2;
        }
        if granule > 1 {
            pieces.push(Piece::Decode);
        }
        pieces
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    fn policy(chunk: usize, granule: usize) -> ChunkPolicy {
        ChunkPolicy::new(nz(chunk), nz(granule), nz(1024)).unwrap()
    }

    fn sizes(pieces: &[Piece]) -> Vec<usize> {
        pieces
            .iter()
            .map(|p| match p {
                Piece::Prefill(n) => n.get(),
                Piece::Decode => 0,
            })
            .collect()
    }

    /// SC-007: at granule 1 and chunk 4 an 11-token prompt is exactly [4, 4, 2, 1]; at granule 2, a
    /// 5-token prompt is [4, decode] and a 7-token prompt [4, 2, decode]. Mutation: replace the
    /// descending powers of two with one residual piece ([4, 4, 3]); the exact-sequence assertion fails.
    #[test]
    fn the_tail_decomposes_into_descending_powers_of_two() {
        assert_eq!(sizes(&policy(4, 1).plan(11)), [4, 4, 2, 1]);
        assert_eq!(sizes(&policy(4, 2).plan(5)), [4, 0]);
        assert_eq!(sizes(&policy(4, 2).plan(7)), [4, 2, 0]);
    }

    /// SC-007: every tail length 0..3 at chunk 4 / granule 1 plans only {1, 2, 4}, and the admitted set
    /// is exactly that. Mutation: as above, the tail-3 plan holds a piece of 3.
    #[test]
    fn every_tail_plans_inside_the_admitted_set() {
        let policy = policy(4, 1);
        let admitted: Vec<usize> = sizes(&policy.admitted_pieces());
        assert_eq!(admitted, [4, 1, 2]);
        for tail in 0..4 {
            let plan = policy.plan(8 + tail);
            assert_eq!(
                plan.iter().map(|p| sizes(&[*p])[0]).sum::<usize>(),
                8 + tail
            );
            for piece in plan {
                assert!(
                    policy.admitted_pieces().contains(&piece),
                    "tail {tail}: {piece:?} is outside the admitted set"
                );
            }
        }
    }

    /// An aligned chunk above the trace limit and a misaligned chunk are both refused, with distinct
    /// typed reasons: alignment alone does not admit a shape.
    #[test]
    fn alignment_does_not_admit_an_over_limit_chunk() {
        assert_eq!(
            ChunkPolicy::new(nz(8), nz(2), nz(4)),
            Err(InvalidOptions::ChunkAboveTraceLimit {
                chunk: 8,
                max_tokens: 4
            })
        );
        assert_eq!(
            ChunkPolicy::new(nz(5), nz(2), nz(64)),
            Err(InvalidOptions::ChunkNotGranuleAligned {
                chunk: 5,
                granule: 2
            })
        );
    }
}
