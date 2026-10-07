//! POOT-1033: `Select` is a bitwise select on every backend. Each case carries the output words the
//! select defines, written as literals (never recomputed with a select), so a row asserts the selection
//! itself and not only agreement with the oracle.
//!
//! The F32 cases poison the unselected branch with a NaN or an infinity and put NaN payloads, infinities
//! and `-0.0` in the selected one: the arithmetic form `if_false + cond * (if_true - if_false)` turns the
//! first into NaN and rounds the second. One case runs the select standalone, one inside a fused float
//! region (`Neg(Neg(select))`, which is the identity on bits), so the two lowerings share one expectation.

use poot_executor::Executor;
use poot_graph_ir::op::FusedOp;
use poot_graph_ir::{Builder, OpKind, Slot, TensorType, Traced, UnOp};
use poot_graph_plan::{Submission, Target};
use poot_quant::weights::WeightStore;
use poot_tensor::{DType, HostTensor};
use poot_test_util::StepFixture;
use poot_test_util::weight_map_oracle::oracle_for;

use crate::Fixture;

const QUIET_NAN_PAYLOAD: u32 = 0x7FC0_1234;
const SIGNALING_NAN: u32 = 0x7F80_0001;
const NEGATIVE_NAN_PAYLOAD: u32 = 0xFFC0_0042;
const POSITIVE_INFINITY: u32 = 0x7F80_0000;
const NEGATIVE_INFINITY: u32 = 0xFF80_0000;
const NEGATIVE_ZERO: u32 = 0x8000_0000;

/// Conditions of 1, 0, 2, -1: any nonzero word selects `if_true`.
const CONDITION: [i32; 8] = [1, 0, 2, -1, 0, 1, 0, -1];
const IF_TRUE: [u32; 8] = [
    QUIET_NAN_PAYLOAD,
    1.0f32.to_bits(),
    POSITIVE_INFINITY,
    NEGATIVE_ZERO,
    5.0f32.to_bits(),
    NEGATIVE_INFINITY,
    SIGNALING_NAN,
    NEGATIVE_NAN_PAYLOAD,
];
const IF_FALSE: [u32; 8] = [
    3.0f32.to_bits(),
    SIGNALING_NAN,
    7.0f32.to_bits(),
    NEGATIVE_NAN_PAYLOAD,
    NEGATIVE_ZERO,
    QUIET_NAN_PAYLOAD,
    POSITIVE_INFINITY,
    NEGATIVE_INFINITY,
];
/// The select's output words, by hand: lane 0 `if_true` (a NaN payload), 1 `if_false` (a signaling NaN),
/// 2 `if_true` for cond 2, 3 `if_true` for cond -1 (`-0.0`), 4 `if_false` (`-0.0`), 5 `if_true`, 6
/// `if_false` (`+Inf`), 7 `if_true` (a negative NaN payload).
const EXPECTED_F32: [u32; 8] = [
    QUIET_NAN_PAYLOAD,
    SIGNALING_NAN,
    POSITIVE_INFINITY,
    NEGATIVE_ZERO,
    NEGATIVE_ZERO,
    NEGATIVE_INFINITY,
    POSITIVE_INFINITY,
    NEGATIVE_NAN_PAYLOAD,
];

const I32_CONDITION: [i32; 4] = [1, 0, 2, -1];
/// Branch words exact in F32 (at most 2^24), so the device can read the select back through a `Cast`.
const I32_IF_TRUE: [i32; 4] = [10, 10, 10, 1 << 22];
const I32_IF_FALSE: [i32; 4] = [20, 20, 20, -(1 << 22)];
/// Any nonzero condition word picks `if_true`: 2 and -1 are not 0/1 flags (the arithmetic form gives 0 and
/// `2 * if_false - if_true` there).
const EXPECTED_I32: [f32; 4] = [10.0, 20.0, 10.0, 4_194_304.0];

/// The case names, in the order of [`select_fixtures`].
pub const STANDALONE_F32: &str = "select_f32_standalone_poisoned";
pub const FUSED_F32: &str = "select_f32_fused_region_poisoned";
pub const STANDALONE_I32: &str = "select_i32_nonzero_condition";

fn f32_tensor(bits: &[u32]) -> HostTensor {
    HostTensor::f32(
        vec![bits.len()],
        bits.iter().map(|&word| f32::from_bits(word)).collect(),
    )
}

/// A fixture whose three inputs are runtime slots, fed the same words on each of two steps: the shared CPU
/// oracle binds a constant through the checkpoint dtypes, which have no I32, and a slot keeps the select on
/// the device whatever the planner folds.
fn slot_fixture(
    name: &'static str,
    build: impl FnOnce(&Builder, [Traced; 3]) -> Traced,
    branch: (Slot, DType),
    inputs: [HostTensor; 3],
) -> Fixture {
    let b = Builder::new();
    let n = inputs[0].numel();
    let cond = b.slot_named(Slot::Token, "cond", TensorType::new(vec![n], DType::I32));
    let if_true = b.slot_named(branch.0, "if_true", TensorType::new(vec![n], branch.1));
    let if_false = b.slot_named(branch.0, "if_false", TensorType::new(vec![n], branch.1));
    let out = build(&b, [cond, if_true, if_false]);
    let graph = b.finish(out);
    let step = || -> Vec<StepFixture> {
        [cond, if_true, if_false]
            .into_iter()
            .zip(&inputs)
            .map(|(slot, tensor)| StepFixture {
                key: graph
                    .meta(slot.id)
                    .slot_key()
                    .expect("a slot input has a key")
                    .clone(),
                tensor: tensor.clone(),
            })
            .collect()
    };
    let steps = vec![step(), step()];
    Fixture {
        name,
        graph,
        store: WeightStore::builder().build(),
        steps,
        guarded: false,
    }
}

fn float_select(name: &'static str, fused: bool) -> Fixture {
    slot_fixture(
        name,
        |b, [cond, if_true, if_false]| {
            let selected = b.select(cond, if_true, if_false);
            // `Neg(Neg(x))` flips the sign bit twice: the identity on bits, and enough pointwise neighbours
            // for `fuse` to put the select inside one float region.
            if fused {
                b.unary(UnOp::Neg, b.unary(UnOp::Neg, selected))
            } else {
                selected
            }
        },
        (Slot::Activation, DType::F32),
        [
            HostTensor::i32(vec![CONDITION.len()], CONDITION.to_vec()),
            f32_tensor(&IF_TRUE),
            f32_tensor(&IF_FALSE),
        ],
    )
}

fn i32_select() -> Fixture {
    let n = I32_CONDITION.len();
    slot_fixture(
        STANDALONE_I32,
        |b, [cond, if_true, if_false]| b.cast(b.select(cond, if_true, if_false), DType::F32),
        (Slot::Token, DType::I32),
        [
            HostTensor::i32(vec![n], I32_CONDITION.to_vec()),
            HostTensor::i32(vec![n], I32_IF_TRUE.to_vec()),
            HostTensor::i32(vec![n], I32_IF_FALSE.to_vec()),
        ],
    )
}

/// The three cases: a standalone F32 select, an F32 select inside a fused float region, and an I32 select.
pub fn select_fixtures() -> Vec<Fixture> {
    vec![
        float_select(STANDALONE_F32, false),
        float_select(FUSED_F32, true),
        i32_select(),
    ]
}

/// The fixture's F32 output equals the hand-written select words, bit for bit (the I32 case reads its select
/// back through a `Cast` to F32, since an exact-I32 output has no host readback).
fn check_bit_exact(fixture: &Fixture, who: &str, got: &HostTensor) -> Result<(), String> {
    let bits: Vec<u32> = got
        .as_f32()
        .ok_or_else(|| format!("{}: {who} is not F32", fixture.name))?
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let expected: Vec<u32> = if fixture.name == STANDALONE_I32 {
        EXPECTED_I32.iter().map(|v| v.to_bits()).collect()
    } else {
        EXPECTED_F32.to_vec()
    };
    if bits == expected {
        Ok(())
    } else {
        Err(format!(
            "{}: {who} words {bits:#010x?}, want {expected:#010x?}",
            fixture.name
        ))
    }
}

/// `fixture` compiled for `target` plans its select the way the case says: the standalone cases run a
/// plain `Select` equation, the fused case runs one `Fused` region that holds the select as a step. A row
/// then shows which lowering produced its words.
fn check_lowering(fixture: &Fixture, target: Target) -> Result<(), String> {
    let staged = crate::staged(fixture, target, Submission::Replay);
    let (mut plain, mut fused) = (0, 0);
    for (_, _, program) in staged.stages() {
        for (eqn, _) in program.planned() {
            match &eqn.op {
                OpKind::Select => plain += 1,
                OpKind::Fused(region)
                    if region.steps.iter().any(|step| step.op == FusedOp::Select) =>
                {
                    fused += 1;
                }
                _ => {}
            }
        }
    }
    let expected = if fixture.name == FUSED_F32 {
        (0, 1)
    } else {
        (1, 0)
    };
    if (plain, fused) == expected {
        Ok(())
    } else {
        Err(format!(
            "{}: (plain Select eqns, fused regions holding a Select) is {:?}, want {expected:?}",
            fixture.name,
            (plain, fused)
        ))
    }
}

/// Every case, on the device `exec` drives: planned the way the case says, and the oracle's and the device's
/// output words equal the select's, bit for bit. A case that fails does not stop the others, so one run shows
/// which of the standalone, fused and I32 lowerings is wrong.
pub fn assert_select_is_bitwise(exec: &mut dyn Executor, target: Target) {
    let mut failures = Vec::new();
    for fixture in &select_fixtures() {
        let oracle = oracle_for(&fixture.graph, &fixture.store)(&fixture.steps[0]);
        failures.extend(check_lowering(fixture, target).err());
        failures.extend(check_bit_exact(fixture, "oracle", &oracle).err());
        match crate::run_outputs(exec, target, fixture) {
            Ok(got) => failures.extend(check_bit_exact(fixture, "device", &got[0]).err()),
            Err(error) => failures.push(format!("{}: device run: {error}", fixture.name)),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
