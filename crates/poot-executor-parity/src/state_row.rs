//! POOT-1074: a recurrent state pool addressed by slot, and `reset_state(Row(n))` on it, on whichever
//! device the caller built.
//!
//! The fixture is the shape a batched recurrent model traces: one carried pool `[SLOTS, WIDTH]`, a
//! `[ROWS]` read index (which pool row each step row continues), a `[ROWS]` forward write map (which pool
//! row each step row's result is written to, or -1 for a masked row) and a step that folds an input into
//! the gathered rows
//! (`next = state + x`). A second carried value, a `Positional { axis: 0 }` pool with no row axis, counts
//! the steps taken and rides along in the output: `Row(n)` must leave it alone. The oracle is a host array
//! updated by the same rule, so a pool row the device addressed wrongly, committed when it should not
//! have, or left unzeroed by a row reset, or a counter a row reset zeroed, differs from it. The fixture
//! runs once per element type (F32, and I32 like the PLE token history).

use std::sync::Arc;

use poot_eval::test_support::StepFixture;
use poot_executor::{BindError, ExecError, Executor, NoSync, StateScope};
use poot_graph_ir::{BinOp, Builder, Graph, Scalar, Slot, SlotKey, StateRole, TensorType};
use poot_graph_plan::Target;
use poot_quant::weights::WeightStore;
use poot_tensor::{DType, HostTensor};

use crate::{Fixture, fixture_weights, staged_on};

/// Rows of the carried pool: the sequences a driver could hold open.
const SLOTS: usize = 4;
/// Elements of one pool row.
const WIDTH: usize = 3;
/// Rows of the step: the sequences one step advances (a step may mask some).
const ROWS: usize = 2;
/// Rows of the step counter pool, which has no row axis.
const COUNTER_ROWS: usize = 5;
/// 2^24: above it an F32 cannot hold every integer. The I32 run folds this into its first inputs and
/// subtracts it from the result before the output is cast to F32, so a pool computed in F32 rounds
/// `2^24 + 1` to `2^24` and the result reads 0 where an exact I32 lane reads 1.
const EXACT_BASE: i32 = 16_777_216;

fn fixture(dtype: DType) -> Fixture {
    let b = Builder::for_tests();
    let pool = b
        .state_input(
            "pool",
            TensorType::new(vec![SLOTS, WIDTH], dtype),
            StateRole::Recurrent,
        )
        .unwrap();
    let counter = b
        .state_input(
            "counter",
            TensorType::new(vec![COUNTER_ROWS, WIDTH], dtype),
            StateRole::Positional { axis: 0 },
        )
        .unwrap();
    let read = b
        .slot_named(
            Slot::Activation,
            "state_read",
            TensorType::new(vec![ROWS], DType::I32),
        )
        .unwrap();
    let write = b.write_map("state_write", ROWS, SLOTS).unwrap();
    let x = b
        .slot_named(
            Slot::Activation,
            "x",
            TensorType::new(vec![ROWS, WIDTH], dtype),
        )
        .unwrap();
    let one = match dtype {
        DType::I32 => Scalar::I32(1),
        _ => Scalar::F32(1.0),
    };
    let current = b.gather(pool, 0, read).unwrap();
    let next = b.binary(BinOp::Add, current, x).unwrap();
    let updated = b.scatter_update(pool, next, write).unwrap();
    let counted = b.binary_scalar(BinOp::Add, counter, one).unwrap();
    // The output is the state's own type: an I32 state is read back as I32 words (a cast of carried
    // words to F32 is not provable from the graph, POOT-726). The I32 result is still shifted down by
    // 2^24, so the rows show the small integers the model computes.
    let shown = match dtype {
        DType::I32 => b
            .binary_scalar(BinOp::Sub, next, Scalar::I32(EXACT_BASE))
            .unwrap(),
        _ => next,
    };
    let out = b.concat(0, &[shown, counted]).unwrap();
    let graph: Graph = b
        .finish(poot_graph_ir::Outputs::new(out).with_state(&[(pool, updated), (counter, counted)]))
        .unwrap();
    Fixture {
        name: "state_pool_row_reset",
        graph,
        store: WeightStore::builder().build(),
        steps: Vec::new(),
    }
}

/// The host model: what the device's carried values should hold.
struct Model {
    pool: [[i32; WIDTH]; SLOTS],
    steps: i32,
    /// What the step subtracts from the rows it shows (`EXACT_BASE` for I32).
    shift: i32,
}

impl Model {
    /// One step: row `r` continues pool row `slots[r]` with input `x[r]`; `live[r]` rows commit.
    /// Returns the `[ROWS + COUNTER_ROWS, WIDTH]` output.
    fn step(&mut self, slots: [usize; ROWS], live: [bool; ROWS], x: &[i32]) -> Vec<i32> {
        let mut out = Vec::new();
        let mut next_pool = self.pool;
        for r in 0..ROWS {
            let next: Vec<i32> = (0..WIDTH)
                .map(|w| self.pool[slots[r]][w] + x[r * WIDTH + w])
                .collect();
            if live[r] {
                next_pool[slots[r]].copy_from_slice(&next);
            }
            out.extend(next.iter().map(|v| v - self.shift));
        }
        self.pool = next_pool;
        self.steps += 1;
        out.extend(std::iter::repeat_n(self.steps, COUNTER_ROWS * WIDTH));
        out
    }
}

fn tensor(dtype: DType, shape: Vec<usize>, values: &[i32]) -> HostTensor {
    match dtype {
        DType::I32 => HostTensor::i32(shape, values.to_vec()),
        _ => HostTensor::f32(shape, values.iter().map(|&v| v as f32).collect()),
    }
}

fn inputs(dtype: DType, slots: [usize; ROWS], live: [bool; ROWS], x: &[i32]) -> Vec<StepFixture> {
    let key = |name: &str| SlotKey::new(Slot::Activation, Some(name));
    let write: Vec<i32> = (0..ROWS)
        .map(|r| if live[r] { slots[r] as i32 } else { -1 })
        .collect();
    vec![
        StepFixture {
            key: key("state_read"),
            tensor: HostTensor::i32(vec![ROWS], slots.iter().map(|&s| s as i32).collect()),
        },
        StepFixture {
            key: SlotKey::new(Slot::WriteMap, Some("state_write")),
            tensor: HostTensor::i32(vec![ROWS], write),
        },
        StepFixture {
            key: key("x"),
            tensor: tensor(dtype, vec![ROWS, WIDTH], x),
        },
    ]
}

/// Drive the pool fixture on `exec` once per element type.
pub fn run_state_pool_row_reset(exec: &mut dyn Executor, target: Target) -> Result<(), String> {
    for dtype in [DType::F32, DType::I32] {
        run_dtype(exec, target, dtype).map_err(|e| format!("{dtype:?}: {e}"))?;
    }
    Ok(())
}

/// Rows address their pool rows by slot whatever their order, a masked row commits nothing, `Row(n)` zeroes
/// exactly pool row `n` (and not the counter), and a refused `Row` zeroes nothing.
fn run_dtype(exec: &mut dyn Executor, target: Target, dtype: DType) -> Result<(), String> {
    let fixture = fixture(dtype);
    let exe = exec
        .load_weights(Arc::new(fixture.store.clone()), fixture_weights(&fixture))
        .map_err(|e| format!("load_weights: {e}"))?;
    let entry = exec
        .add_entry(exe, &staged_on(exec, &fixture, target))
        .map_err(|e| format!("add_entry: {e}"))?;
    let mut model = Model {
        pool: [[0; WIDTH]; SLOTS],
        steps: 0,
        shift: if dtype == DType::I32 { EXACT_BASE } else { 0 },
    };
    // The first inputs sit above 2^24 for I32 (odd values there are not F32 integers).
    let base = model.shift;
    let run = |exec: &mut dyn Executor,
               model: &mut Model,
               what: &str,
               slots: [usize; ROWS],
               live: [bool; ROWS],
               x: [i32; ROWS * WIDTH]|
     -> Result<(), String> {
        let want = model.step(slots, live, &x);
        let step = inputs(dtype, slots, live, &x);
        let host = exec
            .step(
                exe,
                entry,
                &crate::BoundStep::new(&fixture.graph, &step)?.inputs(),
                &mut NoSync,
            )
            .map_err(|e| format!("{what}: step: {e}"))?
            .to_host()
            .map_err(|e| format!("{what}: read: {e}"))?;
        let got: Vec<i32> = match dtype {
            DType::I32 => host
                .as_i32()
                .ok_or_else(|| format!("{what}: the output is not i32"))?
                .to_vec(),
            _ => host
                .as_f32()
                .ok_or_else(|| format!("{what}: the output is not f32"))?
                .iter()
                .map(|&v| v as i32)
                .collect(),
        };
        if got != want {
            return Err(format!(
                "{what}: slots {slots:?} live {live:?}: device {got:?} != oracle {want:?}"
            ));
        }
        Ok(())
    };

    run(
        exec,
        &mut model,
        "first touch",
        [2, 0],
        [true, true],
        [1, 2, 3, 4, 5, 6].map(|v| v + base),
    )?;
    // The same two sequences in the other row order continue their own state.
    run(
        exec,
        &mut model,
        "swapped rows",
        [0, 2],
        [true, true],
        [0, 0, 0, -1, -2, -3],
    )?;
    // A masked row computes but commits nothing: slot 1 stays zero, slot 3 takes its result.
    run(
        exec,
        &mut model,
        "masked row",
        [3, 1],
        [true, false],
        [8, 8, 8, 9, 9, 9],
    )?;
    run(
        exec,
        &mut model,
        "after the mask",
        [1, 3],
        [true, true],
        [1, 1, 1, 1, 1, 1],
    )?;

    // A reset of slot 2 restarts that sequence and leaves slots 0, 1 and 3 as they were.
    exec.reset_state(exe, StateScope::Row(2))
        .map_err(|e| format!("reset_state(Row(2)): {e}"))?;
    model.pool[2] = [0; WIDTH];
    run(
        exec,
        &mut model,
        "after Row(2)",
        [0, 2],
        [true, true],
        [2, 2, 2, 2, 2, 2],
    )?;
    run(
        exec,
        &mut model,
        "slot 3 survived",
        [3, 1],
        [true, true],
        [1, 0, 1, 0, 1, 0],
    )?;

    // A row outside the pool is a typed refusal and zeroes nothing.
    match exec.reset_state(exe, StateScope::Row(SLOTS)) {
        Err(ExecError::Bind(error))
            if matches!(
                *error,
                BindError::StateRow {
                    row: SLOTS,
                    rows: SLOTS,
                    ..
                }
            ) => {}
        other => {
            return Err(format!(
                "Row({SLOTS}) must be a StateRow refusal, got {other:?}"
            ));
        }
    }
    run(
        exec,
        &mut model,
        "after a refused Row",
        [0, 2],
        [true, true],
        [0; ROWS * WIDTH],
    )?;
    exec.unload(exe).map_err(|e| format!("unload: {e}"))?;
    Ok(())
}
