//! Ring collectives (card 049a): the reduce-scatter + all-gather algorithms as pure functions with no GPU
//! dependency.
//!
//! These are poot's own ring collectives (own kernels + interconnect, not NCCL/RCCL). The
//! ring is bandwidth-optimal: each rank moves `2*(R-1)/R` of the buffer, independent of world size.
//! `ring_all_reduce_sum` is reduce-scatter then all-gather; `ring_all_gather` is the all-gather half alone.
//!
//! # The pluggable transfer seam
//!
//! The ring logic never touches memory directly: every cross-rank chunk move goes through
//! [`ChunkTransport::move_chunk`]. Here the transport is [`CpuTransport`], a plain `copy_from_slice`. The
//! same ring structure runs over CUDA P2P in `crate::p2p` (`cuMemcpyPeerAsync` after
//! `cuCtxEnablePeerAccess`, with a device-to-host / host-to-device staging fallback where
//! `cuDeviceCanAccessPeer` is false), with `cuEventRecord` / `cuStreamWaitEvent` ordering the peer copy
//! against the accumulate, and the accumulate using poot's `Binary(Add)` kernel instead of the CPU `+=`.
//! Only the transport and the accumulate op change.

/// The cross-rank chunk transfer, made pluggable so the CPU simulation and the real CUDA P2P copy share
/// one ring. `move_chunk` delivers `src_chunk` (owned by `src_rank`) into `dst_staging` (owned by
/// `dst_rank`); on CPU it is a `copy_from_slice`, on GPU a `cuMemcpyPeerAsync`. Implementors may count
/// transfers for the step-count invariant.
pub trait ChunkTransport {
    /// Copy `src_chunk` from `src_rank` into `dst_staging` at `dst_rank`. `src_chunk` and
    /// `dst_staging` always have equal length (possibly zero, for an empty ring chunk).
    fn move_chunk(
        &mut self,
        src_rank: usize,
        dst_rank: usize,
        src_chunk: &[f32],
        dst_staging: &mut [f32],
    );
}

/// CPU transport: an in-process memcpy that also records, per source rank, how many chunk sends it issued,
/// so a test can assert the ring's defining property: `2*(R-1)` transfers per rank for all-reduce, `R-1`
/// for all-gather (vs `R-1` full-buffer sends per rank in a naive all-to-all).
#[derive(Debug, Default, Clone)]
pub struct CpuTransport {
    /// `sends[r]` = number of `move_chunk` calls with `src_rank == r`.
    pub sends: Vec<usize>,
}

impl CpuTransport {
    /// A transport sized for `world_size` ranks with zeroed send counters.
    pub fn new(world_size: usize) -> Self {
        Self {
            sends: vec![0; world_size],
        }
    }
}

impl ChunkTransport for CpuTransport {
    fn move_chunk(
        &mut self,
        src_rank: usize,
        _dst_rank: usize,
        src_chunk: &[f32],
        dst_staging: &mut [f32],
    ) {
        debug_assert_eq!(
            src_chunk.len(),
            dst_staging.len(),
            "ring chunk length mismatch"
        );
        dst_staging.copy_from_slice(src_chunk);
        if src_rank >= self.sends.len() {
            self.sends.resize(src_rank + 1, 0);
        }
        self.sends[src_rank] += 1;
    }
}

/// Half-open element range `[start, end)` of chunk `i` when a length-`len` buffer is split into `n`
/// contiguous chunks. When `len` is not divisible by `n` the first `len % n` chunks get one extra element,
/// so every element belongs to exactly one chunk. Chunks may be empty when `len < n` (a no-op transfer).
pub(crate) fn chunk_range(len: usize, n: usize, i: usize) -> (usize, usize) {
    let base = len / n;
    let rem = len % n;
    // chunk k has size base + (k < rem), offset = k*base + min(k, rem).
    let start = i * base + i.min(rem);
    let size = base + usize::from(i < rem);
    (start, start + size)
}

/// In-place ring all-reduce with `op = Sum` over `bufs`, one buffer per rank, all of equal length. After
/// the call every buffer holds the elementwise sum of all input buffers. Uses the default [`CpuTransport`];
/// see [`ring_all_reduce_sum_with`] to plug a transport.
///
/// Panics if the buffers are not all the same length.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-749 (formerly 585a)")
)]
pub(crate) fn ring_all_reduce_sum(bufs: &mut [Vec<f32>]) {
    let mut t = CpuTransport::new(bufs.len());
    ring_all_reduce_sum_with(bufs, &mut t);
}

/// [`ring_all_reduce_sum`] with an explicit transport. The reduce-scatter accumulate is a CPU `+=`.
pub(crate) fn ring_all_reduce_sum_with<T: ChunkTransport + ?Sized>(
    bufs: &mut [Vec<f32>],
    transport: &mut T,
) {
    let n = bufs.len();
    if n == 0 {
        return;
    }
    let len = bufs[0].len();
    assert!(
        bufs.iter().all(|b| b.len() == len),
        "ring_all_reduce_sum: all rank buffers must have equal length"
    );
    if n == 1 {
        // One replica: AllReduce(x) = x. No transfers.
        return;
    }

    // Per-destination staging buffer, sized to the largest chunk so one alloc serves all steps.
    let max_chunk = chunk_range(len, n, 0).1 - chunk_range(len, n, 0).0;
    let mut staging: Vec<Vec<f32>> = vec![vec![0.0f32; max_chunk]; n];

    // --- Reduce-scatter: R-1 steps. At step s, rank r sends chunk (r - s) to r+1 and accumulates the
    //     chunk it receives; after the phase rank r fully owns the reduced chunk (r + 1) mod n. ---
    for step in 0..n - 1 {
        // Snapshot all sends first (synchronous ring barrier): each destination pulls its left neighbour's
        // outgoing chunk into its own staging before any accumulate mutates buffers.
        for (dst, dst_staging) in staging.iter_mut().enumerate() {
            let src = (dst + n - 1) % n;
            let send_idx = (src + n - step) % n; // == recv chunk index for dst
            let (a, b) = chunk_range(len, n, send_idx);
            let stg = &mut dst_staging[..b - a];
            transport.move_chunk(src, dst, &bufs[src][a..b], stg);
        }
        // Accumulate the received chunk into the destination's buffer.
        for dst in 0..n {
            let recv_idx = (dst + n - step - 1) % n;
            let (a, b) = chunk_range(len, n, recv_idx);
            let stg = &staging[dst][..b - a];
            for (x, &y) in bufs[dst][a..b].iter_mut().zip(stg) {
                *x += y;
            }
        }
    }

    // --- All-gather: R-1 steps circulating the reduced chunks (overwrite, no add) so every rank ends
    //     with all n reduced chunks. ---
    for step in 0..n - 1 {
        for (dst, dst_staging) in staging.iter_mut().enumerate() {
            let src = (dst + n - 1) % n;
            let send_idx = (src + 1 + n - step) % n; // src's currently-complete chunk this step
            let (a, b) = chunk_range(len, n, send_idx);
            let stg = &mut dst_staging[..b - a];
            transport.move_chunk(src, dst, &bufs[src][a..b], stg);
        }
        for dst in 0..n {
            let recv_idx = (dst + n - step) % n;
            let (a, b) = chunk_range(len, n, recv_idx);
            let stg = &staging[dst][..b - a];
            bufs[dst][a..b].copy_from_slice(stg);
        }
    }
}

/// Ring all-gather: each rank starts with its own `shard`; every rank ends holding the concatenation of
/// all shards in rank order. Returns one gathered buffer per rank (all equal). Shards may have different
/// lengths. Uses the default [`CpuTransport`]; see [`ring_all_gather_with`].
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-749 (formerly 585a)")
)]
pub(crate) fn ring_all_gather(shards: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut t = CpuTransport::new(shards.len());
    ring_all_gather_with(shards, &mut t)
}

/// [`ring_all_gather`] with an explicit transport. `R-1` sends per rank.
pub(crate) fn ring_all_gather_with<T: ChunkTransport + ?Sized>(
    shards: &[Vec<f32>],
    transport: &mut T,
) -> Vec<Vec<f32>> {
    let n = shards.len();
    if n == 0 {
        return Vec::new();
    }

    // result[r][k] = rank r's copy of shard k; initially only slot r is filled.
    let mut slots: Vec<Vec<Vec<f32>>> = (0..n)
        .map(|r| {
            let mut row = vec![Vec::new(); n];
            row[r] = shards[r].clone();
            row
        })
        .collect();

    // R-1 ring steps: at step s, rank r sends slot (r - s) to r+1 and receives slot (r-1-s).
    for step in 0..n.saturating_sub(1) {
        // Snapshot sends first (staging holds each destination's incoming slot).
        let mut staging: Vec<Vec<f32>> = vec![Vec::new(); n];
        for (dst, dst_staging) in staging.iter_mut().enumerate() {
            let src = (dst + n - 1) % n;
            let send_slot = (src + n - step) % n;
            let payload = &slots[src][send_slot];
            let mut stg = vec![0.0f32; payload.len()];
            transport.move_chunk(src, dst, payload, &mut stg);
            *dst_staging = stg;
        }
        for dst in 0..n {
            let recv_slot = (dst + n - step - 1) % n;
            slots[dst][recv_slot] = std::mem::take(&mut staging[dst]);
        }
    }

    // Each rank concatenates its N slots in rank order.
    slots
        .into_iter()
        .map(|row| row.into_iter().flatten().collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64 -> pseudo-random f32 in roughly [-1, 1).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_f32(&mut self) -> f32 {
            // 24 random mantissa bits -> [0,1), then shift to [-1,1).
            let bits = (self.next_u64() >> 40) as u32; // 24 bits
            (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        }
        fn next_small_int(&mut self) -> f32 {
            // Small integer values so the sum is representable exactly in f32 (bit-exact check).
            ((self.next_u64() % 2001) as i64 - 1000) as f32
        }
        fn next_len(&mut self, max: usize) -> usize {
            (self.next_u64() as usize % max) + 1
        }
    }

    fn naive_sum(bufs: &[Vec<f32>]) -> Vec<f32> {
        let len = bufs[0].len();
        let mut out = vec![0.0f32; len];
        for b in bufs {
            for (o, &x) in out.iter_mut().zip(b) {
                *o += x;
            }
        }
        out
    }

    fn naive_concat(shards: &[Vec<f32>]) -> Vec<f32> {
        shards.iter().flatten().copied().collect()
    }

    #[test]
    fn chunk_range_partitions_exactly() {
        for &len in &[0usize, 1, 3, 7, 8, 16, 17, 100, 1000] {
            for &n in &[1usize, 2, 3, 4, 8] {
                let mut covered = 0;
                let mut prev_end = 0;
                for i in 0..n {
                    let (a, b) = chunk_range(len, n, i);
                    assert_eq!(
                        a, prev_end,
                        "chunks must be contiguous (len={len}, n={n}, i={i})"
                    );
                    assert!(b >= a);
                    covered += b - a;
                    prev_end = b;
                }
                assert_eq!(
                    covered, len,
                    "chunks must cover the whole buffer (len={len}, n={n})"
                );
                assert_eq!(
                    prev_end, len,
                    "last chunk must end at len (len={len}, n={n})"
                );
            }
        }
    }

    #[test]
    fn all_reduce_sum_matches_naive_random_f32() {
        let mut rng = Rng::new(0xC0FFEE);
        let mut cases = 0;
        for &n in &[2usize, 3, 4, 8] {
            for _ in 0..8 {
                let len = rng.next_len(300);
                let bufs: Vec<Vec<f32>> = (0..n)
                    .map(|_| (0..len).map(|_| rng.next_f32()).collect())
                    .collect();
                let expected = naive_sum(&bufs);
                let mut got = bufs.clone();
                ring_all_reduce_sum(&mut got);
                for (r, b) in got.iter().enumerate() {
                    assert_eq!(b.len(), len);
                    for (i, (&g, &e)) in b.iter().zip(&expected).enumerate() {
                        assert!(
                            (g - e).abs() <= 1e-4 * (1.0 + e.abs()),
                            "n={n} rank={r} idx={i}: got {g} expected {e}"
                        );
                    }
                }
                cases += 1;
            }
        }
        assert!(cases >= 30, "expected ~30 cases, ran {cases}");
    }

    #[test]
    fn all_reduce_sum_bit_exact_integer_values() {
        // Integer-valued inputs whose sums fit exactly in f32 -> the ring must be bit-exact vs naive.
        let mut rng = Rng::new(0x1234_5678);
        for &n in &[2usize, 3, 4, 8] {
            for _ in 0..5 {
                let len = rng.next_len(200);
                let bufs: Vec<Vec<f32>> = (0..n)
                    .map(|_| (0..len).map(|_| rng.next_small_int()).collect())
                    .collect();
                let expected = naive_sum(&bufs);
                let mut got = bufs.clone();
                ring_all_reduce_sum(&mut got);
                for (r, b) in got.iter().enumerate() {
                    assert_eq!(
                        b, &expected,
                        "n={n} rank={r}: ring != naive sum (integer-valued, bit-exact)"
                    );
                }
            }
        }
    }

    #[test]
    fn all_reduce_start_offset_independent() {
        // Property from the spec: result independent of ring start offset. Rotating the rank order
        // must give the (rotation of the) same all-reduce result - every rank still holds the sum.
        let mut rng = Rng::new(0xABCD_EF01);
        let n = 4;
        let len = 53;
        let bufs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..len).map(|_| rng.next_small_int()).collect())
            .collect();
        let expected = naive_sum(&bufs);
        for rot in 0..n {
            let mut rotated: Vec<Vec<f32>> = (0..n).map(|r| bufs[(r + rot) % n].clone()).collect();
            ring_all_reduce_sum(&mut rotated);
            for b in &rotated {
                assert_eq!(
                    b, &expected,
                    "rotation {rot}: all-reduce sum must be start-offset independent"
                );
            }
        }
    }

    #[test]
    fn all_gather_matches_naive_concat() {
        let mut rng = Rng::new(0xFEED_FACE);
        let mut cases = 0;
        for &n in &[2usize, 3, 4, 8] {
            for _ in 0..8 {
                // Uneven shard lengths across ranks (AllGather does not require equal shards).
                let shards: Vec<Vec<f32>> = (0..n)
                    .map(|_| (0..rng.next_len(50)).map(|_| rng.next_f32()).collect())
                    .collect();
                let expected = naive_concat(&shards);
                let gathered = ring_all_gather(&shards);
                assert_eq!(gathered.len(), n);
                for (r, g) in gathered.iter().enumerate() {
                    assert_eq!(g, &expected, "n={n} rank={r}: all-gather != naive concat");
                }
                cases += 1;
            }
        }
        assert!(cases >= 30, "expected ~30 cases, ran {cases}");
    }

    #[test]
    fn all_reduce_step_count_is_two_r_minus_one_per_rank() {
        // The ring's whole point: 2*(R-1) chunk transfers per rank, not O(R) full buffers (naive
        // all-to-all). Assert per-rank AND total send counts.
        for &n in &[2usize, 3, 4, 8] {
            let len = 64;
            let mut bufs: Vec<Vec<f32>> = (0..n).map(|r| vec![r as f32; len]).collect();
            let mut t = CpuTransport::new(n);
            ring_all_reduce_sum_with(&mut bufs, &mut t);
            for (r, &s) in t.sends.iter().enumerate() {
                assert_eq!(
                    s,
                    2 * (n - 1),
                    "rank {r}: expected 2*(R-1)={} sends, got {s}",
                    2 * (n - 1)
                );
            }
            assert_eq!(
                t.sends.iter().sum::<usize>(),
                n * 2 * (n - 1),
                "total sends must be n*2*(R-1)"
            );
        }
    }

    #[test]
    fn all_gather_step_count_is_r_minus_one_per_rank() {
        for &n in &[2usize, 3, 4, 8] {
            let shards: Vec<Vec<f32>> = (0..n).map(|r| vec![r as f32; 8]).collect();
            let mut t = CpuTransport::new(n);
            let _ = ring_all_gather_with(&shards, &mut t);
            for (r, &s) in t.sends.iter().enumerate() {
                assert_eq!(
                    s,
                    n - 1,
                    "rank {r}: all-gather expected R-1={} sends, got {s}",
                    n - 1
                );
            }
            assert_eq!(t.sends.iter().sum::<usize>(), n * (n - 1));
        }
    }

    #[test]
    fn all_reduce_uneven_length_not_divisible_by_ranks() {
        // len % n != 0 for every n -> exercises the uneven last/first chunks.
        let mut rng = Rng::new(0x0BADC0DE);
        for &n in &[3usize, 4, 8] {
            let len = 7 * n + 1; // guaranteed not divisible by n
            let bufs: Vec<Vec<f32>> = (0..n)
                .map(|_| (0..len).map(|_| rng.next_small_int()).collect())
                .collect();
            let expected = naive_sum(&bufs);
            let mut got = bufs.clone();
            ring_all_reduce_sum(&mut got);
            for b in &got {
                assert_eq!(
                    b, &expected,
                    "n={n} len={len}: uneven-chunk all-reduce != naive sum"
                );
            }
        }
    }

    /// Per-rank output-feature shards gathered over the ring equal the full concatenation in rank order
    /// (the column-parallel case: `ring_all_gather` moved from `tests/ring_in_context.rs`, card 622).
    #[test]
    fn ring_all_gather_reproduces_column_parallel_concat() {
        let mut rng = Rng::new(0x0492_6ACE);
        for &world_size in &[2usize, 3, 4] {
            let shard_n = 5;
            // rank r holds Y[:, r*shard_n:(r+1)*shard_n]; here each shard is just its own slice of data.
            let shards: Vec<Vec<f32>> = (0..world_size)
                .map(|_| (0..shard_n).map(|_| rng.next_f32()).collect())
                .collect();
            let dense: Vec<f32> = shards.iter().flatten().copied().collect();
            let gathered = ring_all_gather(&shards);
            assert_eq!(gathered.len(), world_size);
            for (r, g) in gathered.iter().enumerate() {
                assert_eq!(
                    g, &dense,
                    "R={world_size} rank {r}: ring all-gather != column-parallel concat"
                );
            }
        }
    }

    #[test]
    fn edge_cases_world_size_one_and_len_smaller_than_ranks() {
        // world_size = 1: identity, no panic, no transfers.
        let mut one = vec![vec![1.0f32, 2.0, 3.0]];
        ring_all_reduce_sum(&mut one);
        assert_eq!(one, vec![vec![1.0, 2.0, 3.0]]);

        let g = ring_all_gather(&[vec![9.0f32, 8.0]]);
        assert_eq!(g, vec![vec![9.0, 8.0]]);

        // len < n: some ring chunks are empty; result must still equal the naive sum.
        let n = 8;
        let bufs: Vec<Vec<f32>> = (0..n)
            .map(|r| vec![r as f32, (r * 2) as f32, (r * 3) as f32])
            .collect();
        let expected = naive_sum(&bufs);
        let mut got = bufs.clone();
        ring_all_reduce_sum(&mut got);
        for b in &got {
            assert_eq!(
                b, &expected,
                "len < world_size: empty-chunk ring != naive sum"
            );
        }
    }
}
