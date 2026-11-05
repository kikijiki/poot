//! POOT-1455: the f16 range of attention on a device. Under the narrow policy an attention over an f16 cache rounds
//! its query to f16 row-scaled, `q / s` with `s = max(rowmax(|q|) / 65504, 1)`, and multiplies each row's raw
//! `Q K^T` products by `s` before the score scale (`receipts/1455-attention-f16-range/design.md`). The keys and
//! values are stored f16, so they enter exactly.
//!
//! The fixtures are `precision::attention_fixture`'s: a decode row against a 64-key cache, a 33-row causal prefill,
//! and a 160-row prefill of 16 heads whose rows fill a fragment attention where the device has a matrix unit. Past
//! the range the query reaches 1e6 on every row and the score scale is 1e-6, so the scores stay of order one: the
//! softmax is not one-hot, and the rows hold the device to the oracle's output, not to one argmax.

use poot_graph_ir::{FragmentDtype, MatmulPrecision, OpKind, OperandPrecision, QueryScale};
use poot_tensor::HostTensor;

use crate::Fixture;
use crate::precision::{FRAGMENT_HEADS, Heads, SMALL_HEADS, attention_fixture};

const F16: OperandPrecision = OperandPrecision::Narrow(FragmentDtype::F16);

/// The three shapes, named with `prefix`, over a query of magnitude up to `q_scale / 2`, keys and values up to
/// `kv_scale / 2`, and the scores multiplied by `score_scale`; `precision` is the op's own.
fn fixtures(
    prefix: &'static [&'static str; 3],
    precision: OperandPrecision,
    (q_scale, kv_scale, score_scale): (f32, f32, f32),
) -> Vec<Fixture> {
    let shapes: [(Heads, usize, usize); 3] = [
        (SMALL_HEADS, 1, 64),
        (SMALL_HEADS, 33, 33),
        (FRAGMENT_HEADS, 160, 160),
    ];
    prefix
        .iter()
        .zip(shapes)
        .map(|(&name, (heads, lq, lk))| {
            attention_fixture(
                name,
                precision,
                heads,
                (lq, lk),
                (q_scale, kv_scale),
                score_scale,
            )
        })
        .collect()
}

/// Every query row past f16's range (`|q|` up to 1e6), the policy choosing the precision.
pub fn past_range_fixtures() -> Vec<Fixture> {
    fixtures(
        &[
            "attention_f16_guard_past_range_decode",
            "attention_f16_guard_past_range_prefill",
            "attention_f16_guard_past_range_fragment",
        ],
        OperandPrecision::Exact,
        (2.0e6, 2.0, 1.0e-6),
    )
}

/// The attentions in f16's range, guarded (the policy chooses), beside the same attentions declaring `Narrow(f16)`
/// themselves (a declared precision stands, so they round unscaled: the f16 program without the guard).
pub fn in_range_pairs() -> Vec<(Fixture, Fixture)> {
    let scales = (40.0, 4.0, 0.25);
    let guarded = fixtures(
        &[
            "attention_f16_guard_in_range_decode",
            "attention_f16_guard_in_range_prefill",
            "attention_f16_guard_in_range_fragment",
        ],
        OperandPrecision::Exact,
        scales,
    );
    let unguarded = fixtures(
        &[
            "attention_f16_unguarded_in_range_decode",
            "attention_f16_unguarded_in_range_prefill",
            "attention_f16_unguarded_in_range_fragment",
        ],
        F16,
        scales,
    );
    guarded.into_iter().zip(unguarded).collect()
}

/// The fixture's attention, compiled under the narrow policy for the device `exec` drives, rounds to f16 and reads
/// its query as `query` says (so the device's own f16 unit and the scale are what the row exercises).
fn assert_f16_attention(
    exec: &dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
    fixture: &Fixture,
    query: QueryScale,
) -> Result<(), String> {
    let staged = crate::staged_under(exec, fixture, target, MatmulPrecision::Narrow);
    let found: Vec<(OperandPrecision, QueryScale)> = staged
        .stages()
        .flat_map(|(_, _, program)| {
            program
                .planned()
                .filter_map(|(eqn, _)| match &eqn.op {
                    OpKind::Attention(attrs) => Some((attrs.precision, attrs.query)),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .collect();
    if found != [(F16, query)] {
        return Err(format!(
            "{}: expected one f16 attention reading its query {query:?}, found {found:?}",
            fixture.name
        ));
    }
    Ok(())
}

/// The guard on a device: every query row past f16's range, the attention compiled under the narrow policy rounds
/// to f16 with its query row-scaled, every step's output is finite, and it matches the CPU oracle of the declared
/// semantics (the guarded graph) within the narrow class's tolerance.
pub fn run_past_range(
    exec: &mut dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
) -> Result<(), String> {
    let fixtures = past_range_fixtures();
    for fixture in &fixtures {
        assert_f16_attention(exec, target, fixture, QueryScale::PerRow)?;
        let outputs = crate::run_outputs_under(exec, target, fixture, MatmulPrecision::Narrow)?;
        for (step, output) in outputs.iter().enumerate() {
            let values = output
                .to_f32()
                .map_err(|e| format!("{}: an output does not read as f32: {e}", fixture.name))?;
            if let Some(i) = values.iter().position(|v| !v.is_finite()) {
                return Err(format!(
                    "{}: step {step}, element {i} is {}",
                    fixture.name, values[i]
                ));
            }
        }
    }
    crate::precision::run_parity_declared(exec, target, &fixtures, MatmulPrecision::Narrow)
}

/// In-range identity on a device: every query row's scale is exactly 1, so the guarded attention's output is bit for
/// bit the unguarded f16 attention's, at every step.
pub fn run_in_range_identity(
    exec: &mut dyn poot_executor::Executor,
    target: poot_graph_plan::Target,
) -> Result<(), String> {
    let bits = |outputs: Vec<HostTensor>| -> Result<Vec<Vec<u32>>, String> {
        outputs
            .iter()
            .map(|tensor| {
                tensor
                    .to_f32()
                    .map(|values| values.iter().map(|v| v.to_bits()).collect())
                    .map_err(|e| format!("an output does not read as f32: {e}"))
            })
            .collect()
    };
    for (guarded, unguarded) in in_range_pairs() {
        assert_f16_attention(exec, target, &guarded, QueryScale::PerRow)?;
        assert_f16_attention(exec, target, &unguarded, QueryScale::Absent)?;
        let got = bits(crate::run_outputs_under(
            exec,
            target,
            &guarded,
            MatmulPrecision::Narrow,
        )?)?;
        let want = bits(crate::run_outputs_under(
            exec,
            target,
            &unguarded,
            MatmulPrecision::Narrow,
        )?)?;
        for (step, (got, want)) in got.iter().zip(&want).enumerate() {
            if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
                return Err(format!(
                    "{}: step {step}, element {i}: {} guarded, {} unguarded",
                    guarded.name,
                    f32::from_bits(got[i]),
                    f32::from_bits(want[i])
                ));
            }
        }
    }
    Ok(())
}
