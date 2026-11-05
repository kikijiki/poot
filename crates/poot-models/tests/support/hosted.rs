//! The CPU oracle's side of a hosted gather: the stored rows a step copies from the checkpoint.
//!
//! `compile_staged` hosts a table only a gather reads, so the stage graphs declare the gathered rows as a
//! planner-owned slot and no table. The executor copies those rows from the checkpoint; this supplies the same
//! rows to the oracle from the uncut graph's own table input, for a dense F32 table, so a test can chain the
//! stage graphs and compare them with the unhosted graph's evaluation.
#![allow(
    dead_code,
    reason = "shared by several test binaries; each uses part of it"
)]

use std::collections::HashMap;

use poot_eval::Value;
use poot_graph_ir::{BoundaryId, Graph, SlotKey, ValueId};
use poot_graph_plan::IndexSource;
use poot_graph_plan::staged::StagedProgram;
use poot_tensor::HostTensor;

/// A supplier of the rows of `program`'s hosted gathers, for [`poot_eval::test_support::stage_chain::chain`]:
/// the ids come from the uncut `graph`'s step input (bind time) or from the boundary the device exported (sync
/// time), the table from the uncut graph's constant of the same name bound in `inputs`.
pub(crate) fn hosted_rows<'a>(
    program: &'a StagedProgram,
    graph: &'a Graph,
    inputs: &'a HashMap<ValueId, Value>,
) -> impl FnMut(&SlotKey, &HashMap<BoundaryId, Value>) -> Option<Value> + 'a {
    move |key, boundaries| {
        let (site, rows) = program.hosted().iter().find_map(|site| {
            site.rows
                .iter()
                .find(|rows| &rows.slot == key)
                .map(|rows| (site, rows))
        })?;
        let ids = match &site.index {
            IndexSource::Step(step) => {
                let id = graph
                    .inputs()
                    .iter()
                    .copied()
                    .find(|&id| graph.meta(id).slot_key() == Some(step))
                    .expect("the uncut graph declares the step input a hosted gather reads");
                inputs[&id].clone()
            }
            IndexSource::Device(boundary) => boundaries[boundary].clone(),
        };
        let ids = ids.as_host().expect("dense ids").as_i32().expect("I32 ids");
        let table = graph
            .inputs()
            .iter()
            .copied()
            .find(|&id| graph.meta(id).name().as_deref() == Some(rows.source.as_str()))
            .expect("the uncut graph declares the hosted table");
        let table_value = inputs[&table].as_host().expect("a dense table");
        let width = graph.aval(table).shape[1];
        let words = table_value.as_f32().expect("an F32 table");
        let mut gathered = Vec::with_capacity(ids.len() * width);
        for &id in ids {
            let row = usize::try_from(id).expect("a non-negative id");
            gathered.extend_from_slice(&words[row * width..(row + 1) * width]);
        }
        Some(Value::from(HostTensor::f32(
            rows.aval.shape.clone(),
            gathered,
        )))
    }
}
