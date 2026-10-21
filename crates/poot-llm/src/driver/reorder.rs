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
    Builder, BuilderError, Graph, Slot, SlotKey, StateRole, TensorType, TraceLimits, Traced,
    ValueId,
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
    /// The decode program carries no state value with a leading row axis.
    #[error("the decode program carries no state with a row axis")]
    NoState,
    /// A carried state input has no name to key a snapshot by.
    #[error("a carried state input at value {value} carries no name")]
    UnnamedState {
        /// The unnamed state input's value id.
        value: ValueId,
    },
    /// A state value's row axis is not the program's row count.
    #[error("state {name:?} has row axis {found:?} where the program's is {rows}")]
    RowAxis {
        /// The state's name.
        name: String,
        /// The row count the program has.
        rows: usize,
        /// The row axis found, `None` when the state has no row axis.
        found: Option<usize>,
    },
    /// The graph builder refused an equation.
    #[error(transparent)]
    Build(#[from] BuilderError),
}

/// One carried state value of a compiled program: what a row graph over it declares.
#[derive(Clone, Debug)]
pub(super) struct CarriedState {
    name: Option<String>,
    aval: TensorType,
    role: StateRole,
    input: ValueId,
}

impl CarriedState {
    /// The state pairs of one compiled stage graph, at the avals the planner chose.
    pub(super) fn of_graph(graph: &Graph) -> impl Iterator<Item = Self> + '_ {
        graph.state_pairs().map(|pair| {
            let meta = graph.meta(pair.input);
            Self {
                name: meta.name().map(std::borrow::Cow::into_owned),
                aval: meta.aval.clone(),
                role: pair.role,
                input: pair.input,
            }
        })
    }
}

/// The reorder graph over the `carried` state of a compiled decode entry: each state value becomes its own rows
/// gathered by `row_index`. The avals are the compiled program's, so a cache whose stored extent the planner chose
/// is moved at that extent. Refused when `carried` is empty or a state value has no leading row axis of the common
/// length.
pub(super) fn reorder_graph(
    carried: &[CarriedState],
    limits: TraceLimits,
) -> Result<Graph, RowGraphError> {
    let b = Builder::new(limits);
    let rows = carried
        .first()
        .and_then(|state| state.aval.shape.first().copied())
        .ok_or(RowGraphError::NoState)?;
    let index = row_index(&b, rows)?;
    let mut moved = Vec::new();
    for carried in carried {
        let name = carried.name.as_deref().ok_or(RowGraphError::UnnamedState {
            value: carried.input,
        })?;
        if carried.aval.shape.first() != Some(&rows) {
            return Err(RowGraphError::RowAxis {
                name: name.to_string(),
                rows,
                found: carried.aval.shape.first().copied(),
            });
        }
        let state = b.state_input(name, carried.aval.clone(), carried.role)?;
        moved.push((state, state_row_gather(&b, state, index)?));
    }
    let Some(&(_, out)) = moved.first() else {
        return Err(RowGraphError::NoState);
    };
    Ok(b.finish(poot_graph_ir::Outputs::new(out).with_state(&moved))?)
}
