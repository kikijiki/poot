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

use poot_graph_ir::{
    Builder, BuilderError, Graph, Slot, SlotKey, TensorType, TraceLimits, Traced, ValueId,
};
use poot_tensor::DType;

/// The tag of the row-index activation slot.
pub(super) const BEAM_ROWS_TAG: &str = "beam_rows";

/// The key the row index binds by.
pub(super) fn beam_rows_slot() -> SlotKey {
    SlotKey::new(Slot::Activation, Some(BEAM_ROWS_TAG))
}

/// Declare the row-index slot: `I32 [rows]`, bound by name as [`beam_rows_slot`].
pub(super) fn row_index(b: &Builder, rows: usize) -> Result<Traced, BuilderError> {
    b.slot_named(
        Slot::Activation,
        BEAM_ROWS_TAG,
        TensorType::new(vec![rows], DType::I32),
    )
}

/// The rows of `source` that `index` names, along the leading (row) axis. A reorder gathers a state
/// value by itself; a snapshot gathers from the concatenation of two state values, so an index at or
/// past the row count reaches the second.
pub(super) fn state_row_gather(
    b: &Builder,
    source: Traced,
    index: Traced,
) -> Result<Traced, BuilderError> {
    b.gather(source, 0, index)
}

/// Why a reorder or snapshot graph over a decode program's carried state cannot be built. The driver
/// checks every state pair's row axis before it reorders; these are the typed causes when that check
/// was not enough.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RowGraphError {
    #[error("the decode program carries no state with a row axis")]
    NoState,
    #[error("a carried state input at value {value} carries no name")]
    UnnamedState { value: ValueId },
    #[error("state {name:?} has row axis {found:?} where the program's is {rows}")]
    RowAxis {
        name: String,
        rows: usize,
        found: Option<usize>,
    },
    #[error(transparent)]
    Build(#[from] BuilderError),
}

/// The reorder graph over `decode`'s state pairs: each state value becomes its own rows gathered by
/// `row_index`. Refused when `decode` carries no state or a state value has no leading row axis of the
/// common length.
pub(super) fn reorder_graph(decode: &Graph, limits: TraceLimits) -> Result<Graph, RowGraphError> {
    let b = Builder::new(limits);
    let mut pairs = decode.state_pairs().peekable();
    let rows = pairs
        .peek()
        .and_then(|pair| decode.aval(pair.input).shape.first().copied())
        .ok_or(RowGraphError::NoState)?;
    let index = row_index(&b, rows)?;
    let mut moved = Vec::new();
    for pair in pairs {
        let meta = decode.meta(pair.input);
        let name = meta
            .name
            .as_deref()
            .ok_or(RowGraphError::UnnamedState { value: pair.input })?;
        if meta.aval.shape.first() != Some(&rows) {
            return Err(RowGraphError::RowAxis {
                name: name.to_string(),
                rows,
                found: meta.aval.shape.first().copied(),
            });
        }
        let state = b.state_input(name, meta.aval.clone(), pair.role)?;
        moved.push((state, state_row_gather(&b, state, index)?));
    }
    let Some(&(_, out)) = moved.first() else {
        return Err(RowGraphError::NoState);
    };
    Ok(b.finish(poot_graph_ir::Outputs::new(out).with_state(&moved))?)
}
