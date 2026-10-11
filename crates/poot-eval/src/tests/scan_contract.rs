//! The recurrence scan composite (POOT-634) through the real Builder, inference, graph outputs, decomposition
//! and evaluator seams, for both families: one typed packed result, `y` and `s_out` projected from it, a
//! decomposition that is the shared primitive chain, and the chunk-by-chunk evaluator.

use std::collections::HashMap;

use poot_graph_ir::ops::{
    gdn_prefill_chunked, lower_triangle, mamba2_ssd_decode, mamba2_ssd_prefill,
};
use poot_graph_ir::{Builder, OpKind, Operand, ScanAttrs, ScanFamily, StateRole, TensorType};
use poot_tensor::HostTensor;
use poot_test_util::assert_close;
use poot_test_util::scan::{ScanFixture, ScanInputs};

use crate::{EvalBudget, EvalError, EvalOptions, Value, eval};

/// Evaluate `fixture`'s graph: `(y, s_out)`.
fn run(fixture: &ScanFixture, inputs: &ScanInputs) -> Result<(Vec<f32>, Vec<f32>), EvalError> {
    let scan = fixture.graph();
    // The fixture crate's `Value` is the dependency copy of this crate's, so bind host tensors here.
    let values = fixture
        .bind(&scan, inputs)
        .into_iter()
        .map(|(id, tensor)| (id, Value::from(tensor)))
        .collect();
    let result = eval(
        &scan.graph,
        &values,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )?;
    let y = result.output.into_host().expect("dense y");
    let state = result.state.into_iter().next().expect("one carried state");
    Ok((
        y.as_f32().unwrap().to_vec(),
        state
            .into_host()
            .expect("dense s_out")
            .as_f32()
            .unwrap()
            .to_vec(),
    ))
}

fn families(len: usize, chunk: usize) -> [ScanFixture; 2] {
    [
        ScanFixture::delta_rule(3, len, 4, chunk),
        ScanFixture::ssd(3, len, 4, 5, chunk),
    ]
}

/// `y` and `s_out` are two distinct values, each a `Slice` of the one packed result at the rows its
/// [`ScanLayout`](poot_graph_ir::ScanLayout) names.
#[test]
fn y_and_s_out_are_distinct_typed_projections_of_one_packed_result() {
    for fixture in families(6, 4) {
        let scan = fixture.graph();
        let scans: Vec<_> = scan
            .graph
            .eqns()
            .iter()
            .filter(|eqn| matches!(eqn.op, OpKind::Scan(_)))
            .collect();
        assert_eq!(scans.len(), 1, "{:?}: one scan equation", fixture.family);
        let OpKind::Scan(attrs) = scans[0].op else {
            unreachable!()
        };
        let types: Vec<_> = scans[0]
            .inputs
            .iter()
            .map(|operand| match operand {
                Operand::Value(id) => scan.graph.aval(*id).clone(),
                Operand::Lit(scalar) => scalar.ty(),
            })
            .collect();
        let layout = attrs.layout(&types).unwrap();
        assert_eq!(scan.graph.aval(scans[0].out).shape, layout.packed_shape());
        let (_, carried) = scan.graph.state().next().unwrap();
        assert_ne!(carried, scan.graph.result(), "y and s_out are two values");
        let rows = |value| {
            let eqn = scan
                .graph
                .eqns()
                .iter()
                .find(|e| e.out == value)
                .expect("produced");
            assert!(
                matches!(eqn.inputs[..], [Operand::Value(id)] if id == scans[0].out),
                "a projection of the scan: {:?}",
                eqn.inputs
            );
            match eqn.op {
                OpKind::Slice { axis, start, end } => (axis, start..end),
                ref other => panic!("a projection is a Slice, got {other:?}"),
            }
        };
        assert_eq!(
            rows(scan.graph.result()),
            (2, layout.y_rows()),
            "{:?}: y rows",
            fixture.family
        );
        assert_eq!(
            rows(carried),
            (2, layout.state_rows()),
            "{:?}: s_out rows",
            fixture.family
        );
    }
}

/// Every output and carried-state element agrees with the independent token-by-token f64 recurrence, across
/// a tail (`L % C != 0`), exact chunks, one short chunk and a nonzero seed state.
#[test]
fn scan_matches_the_independent_recurrence() {
    for (len, chunk) in [(6, 4), (8, 4), (5, 8), (1, 4), (7, 4)] {
        for fixture in families(len, chunk) {
            let inputs = fixture.inputs(11);
            let (y, s_out) = run(&fixture, &inputs).unwrap();
            let (want_y, want_s) = fixture.reference(&inputs);
            assert_close(&y, &want_y, 3e-4);
            assert_close(&s_out, &want_s, 3e-4);
        }
    }
}

/// The delta rule's evaluation is the pre-card primitive chain, bit for bit: one definition, not a copy.
#[test]
fn the_delta_rule_scan_is_bitwise_the_pre_card_chain() {
    let fixture = ScanFixture::delta_rule(2, 7, 4, 4);
    let inputs = fixture.inputs(5);
    let (y, s_out) = run(&fixture, &inputs).unwrap();

    let b = Builder::new();
    let seq = |name: &str, w: usize| b.constant(name, TensorType::f32(vec![1, 2, 7, w]));
    let (q, k, v, g, beta) = (
        seq("q", 4),
        seq("k", 4),
        seq("v", 4),
        seq("g", 1),
        seq("beta", 1),
    );
    let s = b.state_input("s", TensorType::f32(vec![1, 2, 4, 4]), StateRole::Recurrent);
    let ids = [q.id(), k.id(), v.id(), g.id(), beta.id(), s.id()];
    // The tables the component used to build and pass in.
    let (o, s_chain) = gdn_prefill_chunked(
        &b,
        q,
        k,
        v,
        g,
        beta,
        s,
        lower_triangle(&b, 4, 0.0),
        lower_triangle(&b, 4, 1.0),
        4,
    );
    let chain = b
        .finish(poot_graph_ir::Outputs::new(o).with_state(&[(s, s_chain)]))
        .unwrap();
    let shapes = fixture.operand_shapes();
    let values: HashMap<_, _> = ids
        .iter()
        .zip(&shapes)
        .zip(&inputs.operands)
        .map(|((&id, (_, shape)), data)| {
            (
                id,
                Value::from(HostTensor::f32(shape.clone(), data.clone())),
            )
        })
        .collect();
    let result = eval(&chain, &values, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
    let want_y = result.output.into_host().unwrap();
    let want_s = result
        .state
        .into_iter()
        .next()
        .unwrap()
        .into_host()
        .unwrap();
    let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&y), bits(want_y.as_f32().unwrap()), "y");
    assert_eq!(bits(&s_out), bits(want_s.as_f32().unwrap()), "s_out");
}

/// The SSD scan matches the decode loop (`mamba2_ssd_decode` stepped per token) within tier 2, and the
/// `h_in`-seeded whole-sequence prefill: bitwise when the sequence fits one chunk (the same single call),
/// within tolerance when it is chunked.
#[test]
fn the_ssd_scan_matches_the_decode_loop_and_the_seeded_prefill() {
    for (len, chunk) in [(5, 8), (7, 4), (16, 4)] {
        let fixture = ScanFixture::ssd(2, len, 3, 4, chunk);
        let inputs = fixture.inputs(23);
        let (y, s_out) = run(&fixture, &inputs).unwrap();

        // The seeded whole-sequence prefill.
        let b = Builder::new();
        let shapes = fixture.operand_shapes();
        let traced: Vec<_> = shapes[..6]
            .iter()
            .map(|(name, shape)| b.constant(name, TensorType::f32(shape.clone())))
            .collect();
        let h = b.state_input(
            "h",
            TensorType::f32(shapes[6].1.clone()),
            StateRole::Recurrent,
        );
        let tril = b.broadcast(
            b.reshape(lower_triangle(&b, len, 0.0), vec![1, 1, len, len]),
            vec![1, 2, len, len],
        );
        let (py, ph) = mamba2_ssd_prefill(
            &b, traced[0], traced[1], traced[2], traced[3], traced[4], traced[5], h, tril,
        );
        let ids: Vec<_> = traced.iter().map(|t| t.id()).chain([h.id()]).collect();
        let prefill = b
            .finish(poot_graph_ir::Outputs::new(py).with_state(&[(h, ph)]))
            .unwrap();
        let values: HashMap<_, _> = ids
            .iter()
            .zip(&shapes)
            .zip(&inputs.operands)
            .map(|((&id, (_, shape)), data)| {
                (
                    id,
                    Value::from(HostTensor::f32(shape.clone(), data.clone())),
                )
            })
            .collect();
        let result = eval(&prefill, &values, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
        let want_y = result.output.into_host().unwrap();
        let want_h = result
            .state
            .into_iter()
            .next()
            .unwrap()
            .into_host()
            .unwrap();
        if len <= chunk {
            let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&y),
                bits(want_y.as_f32().unwrap()),
                "L={len}: y is the single call"
            );
            assert_eq!(
                bits(&s_out),
                bits(want_h.as_f32().unwrap()),
                "L={len}: s_out"
            );
        } else {
            assert_close(&y, want_y.as_f32().unwrap(), 3e-4);
            assert_close(&s_out, want_h.as_f32().unwrap(), 3e-4);
        }

        // The decode loop: `mamba2_ssd_decode` stepped per token from the same seed state.
        let step = ScanFixture { len: 1, ..fixture };
        let step_shapes = step.operand_shapes();
        let d = Builder::new();
        let dt: Vec<_> = step_shapes[..6]
            .iter()
            .map(|(name, shape)| d.constant(name, TensorType::f32(shape.clone())))
            .collect();
        let dh = d.state_input(
            "h",
            TensorType::f32(step_shapes[6].1.clone()),
            StateRole::Recurrent,
        );
        let (dy, dh_out) = mamba2_ssd_decode(&d, dt[0], dt[1], dt[2], dt[3], dt[4], dt[5], dh);
        let ids: Vec<_> = dt.iter().map(|t| t.id()).chain([dh.id()]).collect();
        let decode = d
            .finish(poot_graph_ir::Outputs::new(dy).with_state(&[(dh, dh_out)]))
            .unwrap();
        let mut state = inputs.operands[6].clone();
        let mut loop_y = vec![0.0f32; 2 * len * 3];
        for t in 0..len {
            let mut piece = fixture.window(&inputs, t, 1);
            piece.operands[6] = state.clone();
            let values: HashMap<_, _> = ids
                .iter()
                .zip(&step_shapes)
                .zip(&piece.operands)
                .map(|((&id, (_, shape)), data)| {
                    (
                        id,
                        Value::from(HostTensor::f32(shape.clone(), data.clone())),
                    )
                })
                .collect();
            let result = eval(&decode, &values, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
            let yt = result.output.into_host().unwrap();
            for h in 0..2 {
                loop_y[(h * len + t) * 3..(h * len + t + 1) * 3]
                    .copy_from_slice(&yt.as_f32().unwrap()[h * 3..(h + 1) * 3]);
            }
            state = result
                .state
                .into_iter()
                .next()
                .unwrap()
                .into_host()
                .unwrap()
                .as_f32()
                .unwrap()
                .to_vec();
        }
        assert_close(&y, &loop_y, 3e-4);
        assert_close(&s_out, &state, 3e-4);
    }
}

/// Consume `y` of the first half while carrying `s_out` into a second scan over the second half: the
/// continuation equals one scan over the whole sequence (and so the independent recurrence).
#[test]
fn a_continuation_reads_the_committed_state() {
    for whole in [
        ScanFixture::delta_rule(2, 8, 4, 4),
        ScanFixture::ssd(2, 8, 4, 3, 4),
    ] {
        let inputs = whole.inputs(21);
        let (want_y, want_s) = whole.reference(&inputs);
        let half = ScanFixture { len: 4, ..whole };
        let mut first = whole.window(&inputs, 0, 4);
        let (y0, s0) = run(&half, &first).unwrap();
        let mut second = whole.window(&inputs, 4, 4);
        *second.operands.last_mut().unwrap() = s0;
        let (y1, s1) = run(&half, &second).unwrap();
        first.operands.clear();
        let width = whole.width;
        let y: Vec<f32> = (0..whole.heads)
            .flat_map(|h| {
                let row = h * 4 * width;
                y0[row..row + 4 * width]
                    .iter()
                    .chain(&y1[row..row + 4 * width])
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_close(&y, &want_y, 3e-4);
        assert_close(&s1, &want_s, 3e-4);
    }
}

/// The oracle has no sequence-length ceiling: it evaluates the scan one chunk window at a time with the
/// state carried, so a 4096-position prefill (and one ending in a short tail chunk) at the components' chunk
/// of 64 equals the independent recurrence. Dropping the carry between windows cannot pass this: every
/// window after the first would restart from the seed state.
#[test]
fn a_long_prefill_equals_the_independent_recurrence_with_no_length_ceiling() {
    for len in [4096, 4096 + 37] {
        for fixture in [
            ScanFixture::delta_rule(1, len, 4, 64),
            ScanFixture::ssd(1, len, 4, 3, 64),
        ] {
            let inputs = fixture.inputs(17);
            let (y, s_out) = run(&fixture, &inputs).unwrap();
            let (want_y, want_s) = fixture.reference(&inputs);
            assert_close(&y, &want_y, 3e-4);
            assert_close(&s_out, &want_s, 3e-4);
        }
    }
}

/// An ill-typed scan is refused at trace time by the typing rule, naming what is wrong.
#[test]
fn an_ill_typed_scan_is_refused_by_inference() {
    let attrs = ScanAttrs {
        family: ScanFamily::Ssd,
        chunk: 4,
    };
    let f = |shape: &[usize]| TensorType::f32(shape.to_vec());
    let ok = [
        f(&[1, 2, 6, 3]),
        f(&[1, 2, 6, 5]),
        f(&[1, 2, 6, 5]),
        f(&[1, 2, 6, 1]),
        f(&[1, 2, 1, 1]),
        f(&[1, 2, 1, 1]),
        f(&[1, 2, 5, 3]),
    ];
    assert!(attrs.layout(&ok).is_ok());
    let mut wrong_state = ok.clone();
    wrong_state[6] = f(&[1, 2, 4, 3]);
    assert!(
        attrs.layout(&wrong_state).is_err(),
        "the state must be [1,H,N,P]"
    );
    assert!(
        ScanAttrs { chunk: 3, ..attrs }.layout(&ok).is_err(),
        "a power-of-two chunk"
    );
    assert!(attrs.layout(&ok[..6]).is_err(), "seven operands");
}
