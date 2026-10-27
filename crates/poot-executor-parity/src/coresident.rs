//! POOT-1165: executables that share a graph and differ only in constant data stay apart. One engine holds
//! several executables loaded at once over the same traced graph (so the same plan, kernels and artifact), and
//! each must compute from its own constants however the steps interleave.

use std::sync::Arc;

use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::{Executor, NoSync, StepInputs};
use poot_graph_ir::{BinOp, Builder, TensorType};
use poot_graph_plan::Target;

use crate::{Fixture, fixture_weights, staged_on, store_with};

const ROWS: usize = 4;
const COLS: usize = 8;

/// `a + b` over two F32 `[4, 8]` consts, once per `offset`: the graphs are identical and the constants of
/// fixture `i` are the seeded data shifted by `offset[i]`, so no two fixtures' outputs are equal.
pub fn same_graph_different_data_fixtures(offsets: &[f32]) -> Vec<Fixture> {
    offsets
        .iter()
        .map(|&offset| {
            let b = Builder::for_tests();
            let a = b.constant("a", TensorType::f32(vec![ROWS, COLS])).unwrap();
            let c = b.constant("b", TensorType::f32(vec![ROWS, COLS])).unwrap();
            let sum = b.binary(BinOp::Add, a, c).unwrap();
            let graph = b.finish(sum).unwrap();
            let store = store_with(&graph, |_, data| data.iter_mut().for_each(|v| *v += offset));
            Fixture {
                name: "same graph, different data",
                graph,
                store,
                steps: vec![Vec::new()],
            }
        })
        .collect()
}

/// Load every fixture onto `exec` before stepping any, then step them round-robin, forwards then backwards,
/// and require each output to match its own fixture's oracle within 1e-5 (an F32 add is within a few ulps on every
/// backend, and the fixtures' outputs differ by whole units). An executable that returned another's data names itself and the round.
pub fn run_coresident(
    exec: &mut dyn Executor,
    target: Target,
    fixtures: &[Fixture],
) -> Result<(), String> {
    let mut loaded = Vec::with_capacity(fixtures.len());
    for (i, fixture) in fixtures.iter().enumerate() {
        let exe = exec
            .load_weights(Arc::new(fixture.store.clone()), fixture_weights(fixture))
            .map_err(|e| format!("fixture {i}: load_weights: {e}"))?;
        let entry = exec
            .add_entry(exe, &staged_on(exec, fixture, target))
            .map_err(|e| format!("fixture {i}: add_entry: {e}"))?;
        loaded.push((exe, entry));
    }
    let forwards = 0..fixtures.len();
    for (round, order) in [
        forwards.clone().collect::<Vec<_>>(),
        forwards.rev().collect(),
    ]
    .into_iter()
    .enumerate()
    {
        for i in order {
            let (exe, entry) = loaded[i];
            let fixture = &fixtures[i];
            let got = exec
                .step(exe, entry, &StepInputs::new(), &mut NoSync)
                .map_err(|e| format!("fixture {i}: round {round}: step: {e}"))?
                .to_host()
                .map_err(|e| format!("fixture {i}: round {round}: read: {e}"))?;
            let want = oracle_for(&fixture.graph, &fixture.store)(&fixture.steps[0]);
            let (got, want) = (got.to_f32().unwrap(), want.to_f32().unwrap());
            let close = |g: &f32, w: &f32| (g - w).abs() <= 1e-5 * (1.0 + w.abs());
            if !got.iter().zip(want.iter()).all(|(g, w)| close(g, w)) {
                return Err(format!(
                    "fixture {i}: round {round}: the executable did not compute its own constants \
                     (got {:?}, want {:?})",
                    &got[..4],
                    &want[..4],
                ));
            }
        }
    }
    for (i, (exe, _)) in loaded.into_iter().enumerate() {
        exec.unload(exe)
            .map_err(|e| format!("fixture {i}: unload: {e}"))?;
    }
    Ok(())
}
