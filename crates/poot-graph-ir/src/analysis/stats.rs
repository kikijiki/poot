use super::*;
/// Peak transient activation bytes the graph needs, by a liveness sweep over the topologically ordered
/// eqns. Transient values are those produced by an eqn; graph inputs (weights/consts/slots) are
/// persistent and excluded, so this measures the activation working set that fusion and flash
/// attention shrink. Each value is live from its producing eqn until its last use (graph outputs and
/// state writes live to the end); the peak is the max over the schedule of the sum of live transient
/// bytes.
///
/// A static estimate (an intermediate is freed at its last use, one buffer per value, no aliasing or
/// in-place), so it upper-bounds a real allocator and is a faithful relative measure: the
/// `[1,Hq,L,L]` attention scores dominate a materialized prefill (`O(L^2)`), and `flash_attention`
/// removes them (`O(L*D)`). Engine-produced (derived from poot's own graph), no vendor/runtime API.
pub fn peak_transient_bytes(g: &Graph) -> usize {
    let n = g.eqns.len();
    // birth + last-use index per value. Only eqn-produced values are transient; the rest stay at the
    // never-born sentinel and contribute nothing.
    let mut born = vec![false; g.values.len()];
    let mut last_use = vec![0usize; g.values.len()];
    for (i, e) in g.eqns.iter().enumerate() {
        born[e.out] = true;
        last_use[e.out] = i; // dies at birth if never used
    }
    for (i, e) in g.eqns.iter().enumerate() {
        for o in &e.inputs {
            if let Operand::Value(v) = o
                && born[*v]
            {
                last_use[*v] = i;
            }
        }
    }
    // graph output + state writes are live to the end (never freed within the sweep).
    let mut lives_to_end = vec![false; g.values.len()];
    if born[g.output] {
        lives_to_end[g.output] = true;
    }
    for &(_, so) in &g.state {
        if born[so] {
            lives_to_end[so] = true;
        }
    }
    // deaths[i] = transient values whose last use is eqn i (and that do not live to the end).
    let mut deaths: Vec<Vec<ValueId>> = vec![Vec::new(); n.max(1)];
    for v in 0..g.values.len() {
        if born[v] && !lives_to_end[v] {
            deaths[last_use[v]].push(v);
        }
    }
    let bytes = |v: ValueId| -> usize {
        let t = g.aval(v);
        t.numel() * t.dtype.byte_size()
    };
    let mut live = 0usize;
    let mut peak = 0usize;
    for (i, e) in g.eqns.iter().enumerate() {
        live += bytes(e.out); // the output is allocated
        peak = peak.max(live);
        for &v in &deaths[i] {
            live -= bytes(v); // free everything whose last use was this step
        }
    }
    peak
}

/// Static estimate of the GPU dispatch count a graph incurs: the number of eqns that lower to a kernel
/// launch. `Reshape` is a pure buffer-aliasing view (`Plan::Alias`, no dispatch); every other op
/// (pointwise, reduce, matmul, movement kernels, gather/scatter, a fused region, a flash op) is one
/// dispatch. This is the launch-overhead metric: capture/replay collapses per-token host launches to
/// one, fusion merges pointwise chains into a region, and flash attention collapses the ~10-op
/// softmax-attention chain to a single op. Engine-produced and plan-independent (it counts the graph,
/// not a backend's scheduling), so it is a faithful relative measure of what each transform removes.
pub fn dispatch_count(g: &Graph) -> usize {
    g.eqns
        .iter()
        .filter(|e| !matches!(e.op, OpKind::Reshape { .. }))
        .count()
}
