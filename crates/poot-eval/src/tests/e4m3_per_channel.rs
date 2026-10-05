use std::collections::HashMap;
use std::sync::Arc;

use poot_graph_ir::{Builder, OpKind, Operand, Slot, TensorType, Traced, packed_source_constants};
use poot_quant::format::{ScaleEncoding, WeightFormat};
use poot_quant::{OperandRole, PackedComponentRef, PackedPayload, PackedWeight, SourceRole};

use crate::{EvalBudget, EvalOptions, Value, eval};
use poot_tensor::HostTensor;

fn fixture_e4m3_value(byte: u8) -> f32 {
    match byte {
        0x00 => 0.0,
        0x01 => 1.0 / 512.0,
        0x38 => 1.0,
        0x40 => 2.0,
        0x7e => 448.0,
        0x81 => -1.0 / 512.0,
        0xb8 => -1.0,
        0xc0 => -2.0,
        0xfe => -448.0,
        other => panic!("fixture byte 0x{other:02x} needs an independent expected value"),
    }
}

fn dense_oracle(x: &[f32], k: usize, bytes: &[Vec<u8>], scales: &[f32]) -> Vec<f32> {
    let rows = x.len() / k;
    let mut out = vec![0.0; rows * bytes.len()];
    for row in 0..rows {
        for (channel, (weight, scale)) in bytes.iter().zip(scales).enumerate() {
            let mut acc = 0.0;
            for ki in 0..k {
                acc += x[row * k + ki] * fixture_e4m3_value(weight[ki]) * scale;
            }
            out[row * bytes.len() + channel] = acc;
        }
    }
    out
}

/// Build a `PackedContraction` graph over a `WeightFormat::E4m3PerChannel` descriptor (card 545b:
/// the replacement for the deleted `MatMulDequant`/`QuantScheme::E4M3PerChannel`), evaluate it, and
/// compare against `dense_oracle`. Mirrors `contract_through_the_oracle` in
/// `poot-eval/src/tests/packed_dequant.rs`, which drives the same op over other packed formats.
fn run_case(x_shape: Vec<usize>, x_data: Vec<f32>, bytes: Vec<Vec<u8>>, scales: Vec<f32>) {
    let k = *x_shape.last().unwrap();
    let n = bytes.len();
    let descriptor = PackedWeight::try_new(
        WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::F32,
        },
        [n, k],
    )
    .expect("valid E4m3PerChannel descriptor");
    let weight_bytes: Vec<u8> = bytes.iter().flatten().copied().collect();
    let scale_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|scale| scale.to_le_bytes())
        .collect();
    let owner = Arc::new(
        PackedPayload::try_new(
            descriptor,
            [
                (SourceRole::Planar(OperandRole::Codes), weight_bytes.into()),
                (SourceRole::Planar(OperandRole::Scale), scale_bytes.into()),
            ],
        )
        .expect("valid E4m3PerChannel payload"),
    );

    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(x_shape.clone()));
    let mut plan = b.append_plan(0);
    let mut operands = vec![Operand::Value(x.id)];
    let mut source_ids = Vec::new();
    for (name, ty) in packed_source_constants("layer", descriptor) {
        let role = name.role();
        let input = plan
            .input(name, ty, poot_graph_ir::Storage::Const)
            .expect("stage packed source");
        source_ids.push((input.id, role));
        operands.push(Operand::Value(input.id));
    }
    let output = plan
        .equation(
            OpKind::PackedContraction {
                descriptor,
                blocks: 1,
            },
            operands,
        )
        .expect("emit PackedContraction");
    plan.declare_result(output).expect("declare result");
    let mut prepared = b.preflight_append(plan).expect("preflight append");
    let id = b.commit_append(&mut prepared).expect("commit append");
    let graph = b.finish(Traced { id });

    let expected = dense_oracle(&x_data, k, &bytes, &scales);

    let mut inputs = HashMap::from([(x.id, Value::Host(HostTensor::f32(x_shape, x_data)))]);
    for (source_id, role) in &source_ids {
        inputs.insert(
            *source_id,
            Value::Packed(PackedComponentRef::new(Arc::clone(&owner), *role)),
        );
    }

    let Value::Host(got) = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("packed E4M3PerChannel CPU eval")
        .output
    else {
        panic!("packed E4M3PerChannel eval must produce a dense tensor");
    };
    assert_eq!(got.as_f32().unwrap(), expected.as_slice());
}

#[test]
fn packed_e4m3_per_channel_matches_independent_aligned_and_ragged_oracles() {
    let cases = [
        (
            vec![2, 4],
            vec![1.0, -2.0, 0.5, 0.25, -1.0, 3.0, 2.0, -0.5],
            vec![vec![0x38, 0xb8, 0x01, 0x7e], vec![0x81, 0x40, 0xc0, 0xfe]],
            vec![0.5, -0.25],
        ),
        (
            vec![1, 5],
            vec![2.0, -1.0, 0.25, 0.5, -3.0],
            vec![
                vec![0x01, 0xb8, 0x7e, 0x40, 0x81, 0xfe, 0x7e, 0xfe],
                vec![0xfe, 0x38, 0xc0, 0x00, 0x7e, 0x7e, 0xfe, 0x7e],
            ],
            vec![2.0, -0.125],
        ),
    ];
    for (x_shape, x_data, bytes, scales) in cases {
        let k = *x_shape.last().unwrap();
        // The old word-packed `MatMulDequant` carrier padded each row to a whole number of I32
        // words (`k.div_ceil(4)`); the surviving `PackedContraction`/`E4m3PerChannel` carrier is one
        // raw byte per element with no padding, so trim each row to its logical `k` before packing.
        let bytes = bytes
            .into_iter()
            .map(|row| row[..k].to_vec())
            .collect::<Vec<_>>();
        run_case(x_shape, x_data, bytes, scales);
    }
}

#[test]
fn fixture_detects_lane_zero_and_scale_zero_mutations() {
    let x = [2.0, -1.0, 0.25, 0.5, -3.0];
    let bytes = vec![
        vec![0x01, 0xb8, 0x7e, 0x40, 0x81, 0xfe, 0x7e, 0xfe],
        vec![0xfe, 0x38, 0xc0, 0x00, 0x7e, 0x7e, 0xfe, 0x7e],
    ];
    let scales = [2.0, -0.125];
    let expected = dense_oracle(&x, 5, &bytes, &scales);

    let lane_zero_bytes = bytes
        .iter()
        .map(|row| vec![row[0]; row.len()])
        .collect::<Vec<_>>();
    assert_ne!(
        dense_oracle(&x, 5, &lane_zero_bytes, &scales),
        expected,
        "fixture must catch an implementation that always extracts lane zero"
    );
    assert_ne!(
        dense_oracle(&x, 5, &bytes, &[scales[0], scales[0]]),
        expected,
        "fixture must catch an implementation that always uses scale row zero"
    );
}
