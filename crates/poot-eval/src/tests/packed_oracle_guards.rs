//! Packed-dequant guards that drive poot-eval's crate-private `ops` seams directly (the overflow guard
//! and the by-role component binder): they live with the evaluator, not with the pass-soundness suite
//! in `poot-graph-plan/tests`, which cannot reach `pub(crate)` items.

use std::sync::Arc;

use poot_graph_ir::{Builder, PackedSourceName, TensorType, packed_source_constants};
use poot_quant::format::{ScaleEncoding, WeightFormat};
use poot_quant::{OperandRole, PackedComponentRef, PackedPayload, PackedWeight, SourceRole};

use crate::{EvalError, PackedEvalError, Value};

fn scale_elements(descriptor: PackedWeight) -> usize {
    let bytes_per_element = descriptor
        .format()
        .descriptor()
        .planar_operand(OperandRole::Scale)
        .expect("a packed weight has a Scale operand")
        .element_bytes();
    descriptor.source_bytes(SourceRole::Planar(OperandRole::Scale)) / bytes_per_element
}

fn payload(format: WeightFormat, shape: [usize; 2]) -> Arc<PackedPayload> {
    let descriptor = PackedWeight::try_new(format, shape).unwrap();
    let weight_row_bytes = descriptor.source_shape(SourceRole::Planar(OperandRole::Codes))[1];
    let mut weights = vec![0u8; descriptor.source_bytes(SourceRole::Planar(OperandRole::Codes))];
    match format {
        WeightFormat::E2m1Row32 => {
            for row in 0..shape[0] {
                for byte in 0..weight_row_bytes {
                    weights[row * weight_row_bytes + byte] = (((2 * byte + row + 8) % 16) as u8)
                        << 4
                        | ((2 * byte + row + 1) % 16) as u8;
                }
                if !shape[1].is_multiple_of(2) {
                    let last = (row + 1) * weight_row_bytes - 1;
                    weights[last] &= 0x0f;
                }
            }
        }
        _ => {
            let codes = [0x07, 0x08, 0x00, 0x38, 0xb8, 0x20, 0x40, 0x01];
            for (index, byte) in weights.iter_mut().enumerate() {
                *byte = codes[index % codes.len()];
            }
        }
    }
    let elements = scale_elements(descriptor);
    let scales: Vec<u8> = match format {
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16,
        } => (0..elements)
            .flat_map(|index| [0x3f80u16, 0x4000, 0x3f00, 0x4080][index % 4].to_le_bytes())
            .collect(),
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        } => (0..elements)
            .flat_map(|index| [1.0f32, 2.0, 0.5, 4.0][index % 4].to_le_bytes())
            .collect(),
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::E8m0,
        }
        | WeightFormat::E2m1Row32 => (0..elements)
            .map(|index| [0x7f, 0x80, 0x81, 0x7e][index % 4])
            .collect(),
        _ => unreachable!("payload() fixture covers only the four packed-linear cells"),
    };
    Arc::new(
        PackedPayload::try_new(
            descriptor,
            [
                (SourceRole::Planar(OperandRole::Codes), weights.into()),
                (SourceRole::Planar(OperandRole::Scale), scales.into()),
            ],
        )
        .unwrap(),
    )
}

fn primitive_graph(owner: &Arc<PackedPayload>) -> (poot_graph_ir::Graph, usize, usize) {
    let descriptor = owner.weight();
    let b = Builder::new();
    let [(weight_name, weight_type), (scale_name, scale_type)] = <[(PackedSourceName, TensorType);
        2]>::try_from(
        packed_source_constants("layer", descriptor),
    )
    .expect("a registered packed-linear format has exactly two sources");
    let weight = b.constant(weight_name.as_str(), weight_type);
    let scale = b.constant(scale_name.as_str(), scale_type);
    let output = b.packed_dequant(&[weight, scale], descriptor);
    (b.finish(output), weight.id, scale.id)
}

/// `packed_components` binds each graph operand to `descriptor.sources()[i]`'s own role (the
/// the evaluator binds sources by role); a `PackedComponentRef` whose role
/// disagrees with the position it is bound to is a typed rejection, naming the position
/// (`source.role`), not a silent wrong-byte read.
///
/// Mutation (run 2026-09-29): dropped the `component.role() != role` check in `packed_components` ->
/// this test's expected `Binding { field: "source.role" }` failed to match; the swap was instead
/// caught one check later as `Binding { field: "source.byte_length" }`, because `Codes` and `Scale`
/// happen to differ in byte length for this format too. Restored: the role check is still the
/// correct, most specific rejection (and the only one for a role pair that happens to share a byte
/// length), so it stays.
#[test]
fn packed_dequant_binding_rejects_a_swapped_role() {
    let owner = payload(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        [2, 129],
    );
    let (graph, weight_id, scale_id) = primitive_graph(&owner);
    let descriptor = owner.weight();
    // `weight_id` (the `Codes` operand position) is bound to a `Scale`-role component and vice versa.
    // A generic wrong-shape input would also be caught earlier by `eval_value`'s own input-shape gate
    // (`Codes` and `Scale` have different source shapes), so this calls `packed_components` directly -
    // the seam this mutation targets - rather than going through the outer gate a same-shape role swap
    // would slip past.
    let mut env: Vec<Option<Value>> = vec![None; graph.values.len()];
    env[weight_id] = Some(Value::Packed(PackedComponentRef::new(
        Arc::clone(&owner),
        SourceRole::Planar(OperandRole::Scale),
    )));
    env[scale_id] = Some(Value::Packed(PackedComponentRef::new(
        Arc::clone(&owner),
        SourceRole::Planar(OperandRole::Codes),
    )));
    let result = crate::ops::packed_components(&graph, &graph.eqns[0], &env, descriptor);
    assert!(
        matches!(
            result,
            Err(EvalError::Packed(PackedEvalError::Binding {
                field: "source.role"
            }))
        ),
        "{result:?}"
    );
}
