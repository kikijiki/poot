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
//! whose state row `i` takes. Snapshots reuse this builder.

use poot_graph_ir::{Builder, Graph, Slot, SlotKey, TensorType, ValidationOutputs};
use poot_tensor::DType;

/// The tag of the row-index activation slot.
pub(super) const BEAM_ROWS_TAG: &str = "beam_rows";

/// The key the row index binds by.
pub(super) fn beam_rows_slot() -> SlotKey {
    SlotKey::new(Slot::Activation, Some(BEAM_ROWS_TAG))
}

/// The reorder graph over `decode`'s state pairs: each state value becomes its own rows gathered by
/// `row_index`. Panics if `decode` carries no state or a state value has no leading row axis of the
/// common length; the driver checks the row axis of every state pair before it reorders.
pub(super) fn reorder_graph(decode: &Graph<ValidationOutputs>) -> Graph<ValidationOutputs> {
    let b = Builder::new();
    let mut pairs = decode.state_pairs().peekable();
    let rows = pairs
        .peek()
        .and_then(|pair| decode.aval(pair.input).shape.first().copied())
        .expect("a decode program carries state with a row axis");
    let index = b.slot_named(
        Slot::Activation,
        BEAM_ROWS_TAG,
        TensorType::new(vec![rows], DType::I32),
    );
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
        moved.push((state, b.gather(state, 0, index)));
    }
    let (_, out) = moved[0];
    b.finish_with_state(out, &moved)
        .with_validations(Vec::new())
}
