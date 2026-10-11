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

use poot_graph_ir::{Builder, Graph, StateRole, TensorType, ValidationOutputs};

use super::reorder::{row_index, state_row_gather};

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
    pub(super) snapshot: Graph<ValidationOutputs>,
    /// Copy the rows `idx` names from the shadow set back into live state.
    pub(super) restore: Graph<ValidationOutputs>,
}

/// The snapshot and restore graphs over the `Recurrent` pairs of `schema` (a decode program), or `None`
/// for a program that carries no recurrent state. Panics if the recurrent values disagree on their row
/// axis; the driver checks every state pair's row axis before it compiles the program.
pub(super) fn snapshot_entries(schema: &Graph<ValidationOutputs>) -> Option<SnapshotEntries> {
    let carried: Vec<Carried> = schema
        .state_pairs()
        .filter(|pair| pair.role == StateRole::Recurrent)
        .map(|pair| {
            let meta = schema.meta(pair.input);
            Carried {
                name: meta.name.clone().expect("a state input carries its name"),
                aval: meta.aval.clone(),
            }
        })
        .collect();
    let rows = carried.first()?.aval.shape.first().copied()?;
    for value in &carried {
        assert_eq!(
            value.aval.shape.first(),
            Some(&rows),
            "state {:?} has a different row axis",
            value.name
        );
    }
    Some(SnapshotEntries {
        snapshot: copy_graph(&carried, rows, Copy::LiveToShadow),
        restore: copy_graph(&carried, rows, Copy::ShadowToLive),
    })
}

/// Which way a copy graph moves rows.
#[derive(Clone, Copy)]
enum Copy {
    LiveToShadow,
    ShadowToLive,
}

fn copy_graph(carried: &[Carried], rows: usize, direction: Copy) -> Graph<ValidationOutputs> {
    let b = Builder::new();
    let index = row_index(&b, rows);
    let mut pairs = Vec::with_capacity(2 * carried.len());
    let mut last = None;
    for value in carried {
        let live = b.state_input(&value.name, value.aval.clone(), StateRole::Recurrent);
        let shadow = b.state_input(
            &shadow_name(&value.name),
            value.aval.clone(),
            StateRole::Recurrent,
        );
        let copied = match direction {
            Copy::LiveToShadow => {
                let copied = state_row_gather(&b, b.concat(0, &[shadow, live]), index);
                pairs.push((live, live));
                pairs.push((shadow, copied));
                copied
            }
            Copy::ShadowToLive => {
                let copied = state_row_gather(&b, b.concat(0, &[live, shadow]), index);
                pairs.push((live, copied));
                pairs.push((shadow, shadow));
                copied
            }
        };
        last = Some(copied);
    }
    b.finish_with_state(last.expect("a recurrent program carries state"), &pairs)
        .with_validations(Vec::new())
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
