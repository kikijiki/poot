use poot_graph_ir::Builder;

/// Shared-pool scatter (spec 045 / 046b): write each row's new K/V `update[b]` into one shared pool
/// `pool[P,Hkv,D]` at that row's global slot `slot_map[b][pos[b]]` (axis 0). All rows write into the same
/// buffer, so the B single-slot `dynamic_update_slice` writes are threaded (each on the prior result)
/// rather than concatenated. `update[B,Hkv,1,D]`, `pos[B]`, `slot_map[B,cap]` (global slots).
/// `pub(crate)`: gemma4's batched shared-pool decode reuses it (the scatter is arch-neutral).
pub fn scatter_shared_pool(
    b: &Builder,
    pool: poot_graph_ir::Traced,
    update: poot_graph_ir::Traced,
    pos: poot_graph_ir::Traced,
    slot_map: poot_graph_ir::Traced,
    batch: usize,
) -> poot_graph_ir::Traced {
    let cap = b.aval(slot_map).shape[1];
    let (hkv, d) = {
        let s = b.aval(pool).shape;
        (s[1], s[2])
    };
    (0..batch).fold(pool, |pool, row| {
        // this row's [1,Hkv,1,D] update -> [1,Hkv,D] to fit the pool's axis-0 slot.
        let update_row = b.slice(update, 0, row, row + 1); // [1, Hkv, 1, D]
        let update_row = b.reshape(update_row, vec![1, hkv, d]); // [1, Hkv, D]
        let pos_row = b.reshape(b.slice(pos, 0, row, row + 1), vec![]); // scalar logical pos
        let sm_row = b.reshape(b.slice(slot_map, 0, row, row + 1), vec![cap]); // this row's [cap] global map
        let phys_row = b.gather_scalar(sm_row, 0, pos_row); // slot_map[row][pos] -> global pool slot (scalar)
        b.dynamic_update_slice_dyn(pool, update_row, phys_row, 0)
    })
}

/// Shared-pool gather (spec 045 / 046b): read each row's `cap` logical K/V slots from the shared pool
/// `pool[P,Hkv,D]` by its global slot map `slot_map[b]` (a `[cap]` axis-0 gather), then assemble
/// `[B,Hkv,cap,D]` in logical order. Unwritten positions point at any in-range slot and are zeroed by the
/// additive mask (as in `gather_per_row`). `pub(crate)`: reused by gemma4's shared-pool decode tracer.
pub fn gather_shared_pool(
    b: &Builder,
    pool: poot_graph_ir::Traced,
    slot_map: poot_graph_ir::Traced,
    batch: usize,
    cap: usize,
    hkv: usize,
    d: usize,
) -> poot_graph_ir::Traced {
    let rows: Vec<poot_graph_ir::Traced> = (0..batch)
        .map(|row| {
            let sm_row = b.reshape(b.slice(slot_map, 0, row, row + 1), vec![cap]); // this row's [cap] map
            let g = b.gather(pool, 0, sm_row); // [cap, Hkv, D] in logical order
            let g = b.reshape(g, vec![1, cap, hkv, d]); // [1, cap, Hkv, D]
            b.transpose(g, vec![0, 2, 1, 3]) // [1, Hkv, cap, D]
        })
        .collect();
    rows.into_iter()
        .reduce(|acc, r| b.concat(0, &[acc, r]))
        .expect("batch >= 1")
}
