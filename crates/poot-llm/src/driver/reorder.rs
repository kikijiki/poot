//! The row reorder: one compiled graph that moves every carried state value to the row its parent
//! occupied (Card 752, dserve.md ruling V1/S4).
//!
//! Every carried state value's leading axis is the row axis, so a reorder is family-neutral: for each
//! state pair of a decode program, `Gather(state, axis 0, row_index)` assigned back to the state. A state
//! transition is a simultaneous assignment, so a row may be read by several others and no row is
//! overwritten before it is read. KV and recurrent state move alike. The graph declares the same state
//! inputs (name and aval) as the decode program, so the executable shares their buffers with it.
//!
//! The index is a named activation slot, [`BEAM_ROWS_TAG`] (I32, `[rows]`): `row_index[i]` is the row
//! whose state row `i` takes. The snapshot and restore entries ([`super::snapshot`]) build on the same
//! [`row_index`] and [`state_row_gather`], so beam reorder and snapshot cannot drift (R-647-2).

use poot_graph_ir::{Builder, Graph, Slot, SlotKey, TensorType, Traced};
use poot_tensor::DType;

/// The tag of the row-index activation slot.
pub(super) const BEAM_ROWS_TAG: &str = "beam_rows";

/// The key the row index binds by.
pub(super) fn beam_rows_slot() -> SlotKey {
    SlotKey::new(Slot::Activation, Some(BEAM_ROWS_TAG))
}

/// Declare the row-index slot: `I32 [rows]`, bound by name as [`beam_rows_slot`].
pub(super) fn row_index(b: &Builder, rows: usize) -> Traced {
    b.slot_named(
        Slot::Activation,
        BEAM_ROWS_TAG,
        TensorType::new(vec![rows], DType::I32),
    )
}

/// The rows of `source` that `index` names, along the leading (row) axis. A reorder gathers a state
/// value by itself; a snapshot gathers from the concatenation of two state values, so an index at or
/// past the row count reaches the second.
pub(super) fn state_row_gather(b: &Builder, source: Traced, index: Traced) -> Traced {
    b.gather(source, 0, index)
}

/// The reorder graph over `decode`'s state pairs: each state value becomes its own rows gathered by
/// `row_index`. Panics if `decode` carries no state or a state value has no leading row axis of the
/// common length; the driver checks the row axis of every state pair before it reorders.
pub(super) fn reorder_graph(decode: &Graph) -> Graph {
    let b = Builder::new();
    let mut pairs = decode.state_pairs().peekable();
    let rows = pairs
        .peek()
        .and_then(|pair| decode.aval(pair.input).shape.first().copied())
        .expect("a decode program carries state with a row axis");
    let index = row_index(&b, rows);
    let mut moved = Vec::new();
    for pair in pairs {
        let meta = decode.meta(pair.input);
        let name = meta
            .name
            .as_deref()
            .expect("a state input carries its name");
        assert_eq!(
            meta.aval.shape.first(),
            Some(&rows),
            "state {name:?} has a different row axis"
        );
        let state = b.state_input(name, meta.aval.clone(), pair.role);
        moved.push((state, state_row_gather(&b, state, index)));
    }
    let (_, out) = moved[0];
    b.finish(poot_graph_ir::Outputs::new(out).with_state(&moved))
        .unwrap()
}
