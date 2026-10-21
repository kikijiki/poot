//! Snapshot and restore of recurrent state: the rollback a speculative round needs on a model whose
//! rejected draft tokens were folded into state (POOT-631's accepted design, POOT-647).
//!
//! A recurrent value cannot be rolled back by position: folding a token in is not invertible. A round
//! therefore copies the rows it verifies into a shadow state set before the window runs, and a partial
//! accept copies them back and replays the kept tokens. Both copies are graphs built from the decode
//! program's state schema, compiled into the executable the rows run on, so they share buffers with the
//! decode entries by name and aval:
//!
//! - snapshot: `s@snap' = Gather(Concat0(s@snap, s), idx)`
//! - restore: `s' = Gather(Concat0(s, s@snap), idx)`
//!
//! `idx` is the row-index activation slot of [`super::reorder`] (I32, `[rows]`): `r` keeps row `r` of the
//! destination, `rows + r` takes row `r` of the source. Only `StateRole::Recurrent` pairs are covered:
//! positional KV needs no restore, since a write beyond the committed position is overwritten or masked.
//! The side a graph does not write is an alias pair `(x, x)`, which validates and copies nothing.

use poot_graph_ir::{Builder, Graph, StateRole, TensorType, TraceLimits};

use super::reorder::{RowGraphError, row_index, state_row_gather};

/// The shadow of the carried state value `name`.
fn shadow_name(name: &str) -> String {
    format!("{name}@snap")
}

/// One carried recurrent value of a decode program.
struct Carried {
    name: String,
    aval: TensorType,
}

/// The two graphs of a round's rollback.
pub(super) struct SnapshotEntries {
    /// Copy the rows `idx` names from live state into the shadow set.
    pub(super) snapshot: Graph,
    /// Copy the rows `idx` names from the shadow set back into live state.
    pub(super) restore: Graph,
}

/// The snapshot and restore graphs over the `Recurrent` pairs of `schema` (a decode program), or `None`
/// for a program that carries no recurrent state. Refused when the recurrent values disagree on their
/// row axis; the driver checks every state pair's row axis before it compiles the program.
pub(super) fn snapshot_entries(
    schema: &Graph,
    limits: TraceLimits,
) -> Result<Option<SnapshotEntries>, RowGraphError> {
    let mut carried: Vec<Carried> = Vec::new();
    for pair in schema
        .state_pairs()
        .filter(|pair| pair.role == StateRole::Recurrent)
    {
        let meta = schema.meta(pair.input);
        let name = meta
            .name()
            .ok_or(RowGraphError::UnnamedState { value: pair.input })?
            .into_owned();
        carried.push(Carried {
            name,
            aval: meta.aval.clone(),
        });
    }
    let Some(rows) = carried
        .first()
        .and_then(|value| value.aval.shape.first().copied())
    else {
        return Ok(None);
    };
    for value in &carried {
        if value.aval.shape.first() != Some(&rows) {
            return Err(RowGraphError::RowAxis {
                name: value.name.clone(),
                rows,
                found: value.aval.shape.first().copied(),
            });
        }
    }
    Ok(Some(SnapshotEntries {
        snapshot: copy_graph(&carried, rows, Copy::LiveToShadow, limits)?,
        restore: copy_graph(&carried, rows, Copy::ShadowToLive, limits)?,
    }))
}

/// Which way a copy graph moves rows.
#[derive(Clone, Copy)]
enum Copy {
    LiveToShadow,
    ShadowToLive,
}

fn copy_graph(
    carried: &[Carried],
    rows: usize,
    direction: Copy,
    limits: TraceLimits,
) -> Result<Graph, RowGraphError> {
    let b = Builder::new(limits);
    let index = row_index(&b, rows)?;
    let mut pairs = Vec::with_capacity(2 * carried.len());
    let mut last = None;
    for value in carried {
        let live = b.state_input(&value.name, value.aval.clone(), StateRole::Recurrent)?;
        let shadow = b.state_input(
            &shadow_name(&value.name),
            value.aval.clone(),
            StateRole::Recurrent,
        )?;
        let copied = match direction {
            Copy::LiveToShadow => {
                let copied = state_row_gather(&b, b.concat(0, &[shadow, live])?, index)?;
                pairs.push((live, live));
                pairs.push((shadow, copied));
                copied
            }
            Copy::ShadowToLive => {
                let copied = state_row_gather(&b, b.concat(0, &[live, shadow])?, index)?;
                pairs.push((live, copied));
                pairs.push((shadow, shadow));
                copied
            }
        };
        last = Some(copied);
    }
    let Some(last) = last else {
        return Err(RowGraphError::NoState);
    };
    Ok(b.finish(poot_graph_ir::Outputs::new(last).with_state(&pairs))?)
}

/// The row index of a copy: `slots` take their row from the source, every other row keeps its own.
/// `rows` is the graph's row axis.
pub(super) fn copy_rows(rows: usize, slots: &[usize]) -> Vec<i32> {
    let mut index: Vec<i32> = (0..rows as i32).collect();
    for &slot in slots {
        index[slot] = (rows + slot) as i32;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `copy_rows` names the source row (`rows + slot`) of a copied slot and keeps every other row.
    #[test]
    fn a_copied_slot_takes_its_source_row_and_the_rest_keep_theirs() {
        assert_eq!(copy_rows(4, &[1, 3]), [0, 5, 2, 7]);
        assert_eq!(copy_rows(3, &[]), [0, 1, 2]);
    }
}
