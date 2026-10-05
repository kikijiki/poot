/// Maximum captured AQL packets that fit in one physical replay batch for a queue of `queue_size` (its
/// returned half-ring capacity). Diagnostic only today (logged at queue creation); the batched-replay
/// admission path that once computed this for real submissions was deleted with the packed block-float
/// typed walk (card 573b).
pub(crate) fn replay_graph_chunk_size(queue_size: u64) -> usize {
    (queue_size / 2).max(1) as usize
}
