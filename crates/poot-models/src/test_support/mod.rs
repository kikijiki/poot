/// The pre-`compile` IR passes alone, in the former `optimize()` pipeline's
/// order (deleted with card 626 - `poot_graph_plan::compile` is the one pipeline now, but it also
/// needs a `Target`, which this crate's own fusion/flash-shape tests never had). Backend-free.
#[cfg(test)]
pub(crate) use poot_graph_plan::passes_without_target as optimize;
