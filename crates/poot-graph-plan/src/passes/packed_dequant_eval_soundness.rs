//! In-crate test module (moved from `tests/`): `reject_preexisting_packed_contractions` is held
//! non-`pub` for Card 672 (dead-pub W10 form), which an integration test could not reach.

use std::collections::HashMap;
use std::sync::Arc;

use super::{
    PackedDequantProductionError, dce_with_roots, recognize_packed_contractions,
    reject_packed_dequant_escapes, reject_preexisting_packed_contractions,
};
use poot_graph_ir::ops::{packed_block_diagonal_linear, packed_linear};
use poot_graph_ir::{
    Builder, BuilderAppendError, Graph, OpKind, Operand, PackedSourceName, Slot, TensorType,
    Traced, packed_source_constants,
};
use poot_quant::format::{ScaleEncoding, Storage, WeightFormat};
use poot_quant::{OperandRole, PackedComponentRef, PackedPayload, PackedWeight, SourceRole};

use poot_eval::{EvalBudget, EvalError, EvalOptions, PackedEvalError, Value, eval};
use poot_tensor::HostTensor;

/// Thread-local allocation byte counter (duplicated from `poot-eval/src/tests/allocation.rs`: an
/// external integration test cannot reach that `pub(super)` helper, and `#[global_allocator]` can
/// only be declared once per binary - this file is its own binary, so one copy here is safe).
mod allocation {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static RECORDING: Cell<bool> = const { Cell::new(false) };
        static BYTES: Cell<usize> = const { Cell::new(0) };
    }

    struct CountingAllocator;

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    fn note_allocation(size: usize) {
        if RECORDING.try_with(Cell::get).unwrap_or(false) {
            let _ = BYTES.try_with(|bytes| bytes.set(bytes.get().saturating_add(size)));
        }
    }

    // SAFETY: every method forwards to `System` with the caller's arguments unchanged and only
    // counts the requested size, so `System`'s allocator contract carries over.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            note_allocation(layout.size());
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            note_allocation(layout.size());
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            note_allocation(new_size);
            unsafe { System.realloc(pointer, layout, new_size) }
        }
    }

    pub(crate) fn allocated_bytes<T>(body: impl FnOnce() -> T) -> (T, usize) {
        BYTES.with(|bytes| bytes.set(0));
        RECORDING.with(|recording| recording.set(true));
        let output = body();
        RECORDING.with(|recording| recording.set(false));
        (output, BYTES.with(Cell::get))
    }
}
use allocation::allocated_bytes;

/// Card 554d FLAG (migration guide constraint 3, packed-oracle analogue): `walk.rs` deleted the fixed
/// crate consts `PACKED_ORACLE_MAX_ELEMENTS`/`PACKED_ORACLE_MAX_BYTES` and `check_packed_oracle_allocation`,
/// since the packed oracle's allocation limit is now an opt-in `EvalBudget` the caller states per call,
/// not a hardcoded wall (see `ops/packed.rs`'s module doc). This constant mirrors the old fixed value
/// purely so this file's still-meaningful fixture ("a table far bigger than the old cap still evaluates
/// uncapped", `a_packed_projection_past_the_oracle_cap_evaluates` below) keeps the same descriptive
/// sanity assertion; it is NOT a reinstated limit and nothing here enforces it. That same function used
/// to have a second half asserting that materializing the same weight past the cap is refused - removed,
/// since the fixed cap it asserted against is gone and the uncapped `EvalBudget` this file states makes
/// that materialization succeed instead; see the FLAG comment on the function itself. The packed
/// oracle's own `OracleArithmeticOverflow` refusal (never a cap, an actual `usize` overflow) is still
/// live and still tested: `packed_dequant_oracle_rejects_an_overflowing_activation_shape` below.
const PACKED_ORACLE_MAX_ELEMENTS: usize = 8 * 1024 * 1024;

/// Validate, eliminate dead oracle-only decoding, recognize dense contractions, then reject every
/// remaining escape: the graph-level packed-dequant preparation the planner's production entry
/// extends with block-float claims.
fn prepare_packed_dequant_production(g: &Graph) -> Result<Graph, PackedDequantProductionError> {
    g.validate()?;
    reject_preexisting_packed_contractions(g)?;
    let graph = dce_with_roots(g, &[]);
    let graph = recognize_packed_contractions(&graph);
    reject_packed_dequant_escapes(&graph)?;
    graph.validate()?;
    Ok(graph)
}

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

fn independent_fp4(owner: &PackedPayload, coordinate: [usize; 2]) -> f32 {
    const VALUES: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let packed_column = coordinate[1] / 2;
    let byte =
        owner.bytes(SourceRole::Planar(OperandRole::Codes))[coordinate[0] * 18 + packed_column];
    let code = if coordinate[1].is_multiple_of(2) {
        byte & 0x0f
    } else {
        byte >> 4
    };
    let scale_column = coordinate[1] / 32;
    let scale =
        owner.bytes(SourceRole::Planar(OperandRole::Scale))[coordinate[0] * 2 + scale_column];
    VALUES[usize::from(code)] * f32::from_bits(u32::from(scale) << 23)
}

fn independent_e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 {
        sign * f32::from(mantissa) * 2.0f32.powi(-9)
    } else {
        sign * (1.0 + f32::from(mantissa) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
    }
}

fn independent_fp8(owner: &PackedPayload, coordinate: [usize; 2]) -> f32 {
    let weight =
        owner.bytes(SourceRole::Planar(OperandRole::Codes))[coordinate[0] * 129 + coordinate[1]];
    let scale_index = (coordinate[0] / 128) * 2 + coordinate[1] / 128;
    let scale_bytes = owner.bytes(SourceRole::Planar(OperandRole::Scale));
    let format = owner.weight().format();
    let scale = match format {
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16,
        } => {
            let offset = scale_index * 2;
            f32::from_bits(
                u32::from(u16::from_le_bytes([
                    scale_bytes[offset],
                    scale_bytes[offset + 1],
                ])) << 16,
            )
        }
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        } => {
            let offset = scale_index * 4;
            f32::from_le_bytes(scale_bytes[offset..offset + 4].try_into().unwrap())
        }
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::E8m0,
        } => f32::from_bits(u32::from(scale_bytes[scale_index]) << 23),
        _ => unreachable!("independent_fp8 covers only the three E4M3 block-128 cells"),
    };
    independent_e4m3(weight) * scale
}

#[test]
fn packed_dequant_eval_matches_independent_rows() {
    for format in [
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::Bf16,
        },
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::E8m0,
        },
        WeightFormat::E2m1Row32,
    ] {
        let shape = if format == WeightFormat::E2m1Row32 {
            [2, 35]
        } else {
            [129, 129]
        };
        let owner = payload(format, shape);
        if format == WeightFormat::E2m1Row32 {
            let mut seen = [false; 16];
            for &byte in owner.bytes(SourceRole::Planar(OperandRole::Codes)) {
                seen[usize::from(byte & 0x0f)] = true;
                seen[usize::from(byte >> 4)] = true;
            }
            assert!(seen[8..].iter().all(|present| *present));
        } else {
            assert_eq!(owner.bytes(SourceRole::Planar(OperandRole::Codes))[0], 0x07);
            assert_eq!(owner.bytes(SourceRole::Planar(OperandRole::Codes))[1], 0x08);
        }
        let (graph, weight, scale) = primitive_graph(&owner);
        let inputs = HashMap::from([
            (
                weight,
                Value::Packed(PackedComponentRef::new(
                    Arc::clone(&owner),
                    SourceRole::Planar(OperandRole::Codes),
                )),
            ),
            (
                scale,
                Value::Packed(PackedComponentRef::new(
                    Arc::clone(&owner),
                    SourceRole::Planar(OperandRole::Scale),
                )),
            ),
        ]);
        let Value::Host(actual) = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| r.output)
            .unwrap()
        else {
            panic!("packed dequant must produce dense oracle output");
        };
        for row in 0..shape[0] {
            for column in 0..shape[1] {
                let expected = if format == WeightFormat::E2m1Row32 {
                    independent_fp4(&owner, [row, column])
                } else {
                    independent_fp8(&owner, [row, column])
                };
                assert_eq!(
                    actual.as_f32().unwrap()[row * shape[1] + column].to_bits(),
                    expected.to_bits(),
                    "{format:?} coordinate [{row},{column}]"
                );
            }
        }
    }
}

#[test]
fn packed_dequant_binding_reuses_payload_owner() {
    let owner = payload(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        [2, 129],
    );
    let weak = Arc::downgrade(&owner);
    let baseline = Arc::strong_count(&owner);
    let weight =
        PackedComponentRef::new(Arc::clone(&owner), SourceRole::Planar(OperandRole::Codes));
    let scale = PackedComponentRef::new(Arc::clone(&owner), SourceRole::Planar(OperandRole::Scale));
    assert!(weight.same_owner(&scale));
    assert!(Arc::ptr_eq(weight.owner(), &owner));
    assert!(Arc::ptr_eq(scale.owner(), &owner));
    assert_eq!(
        weight.bytes().as_ptr(),
        owner.bytes(SourceRole::Planar(OperandRole::Codes)).as_ptr()
    );
    assert_eq!(
        scale.bytes().as_ptr(),
        owner.bytes(SourceRole::Planar(OperandRole::Scale)).as_ptr()
    );
    assert_eq!(Arc::strong_count(&owner), baseline + 2);
    let (graph, weight_id, scale_id) = primitive_graph(&owner);
    let bindings = HashMap::from([
        (weight_id, Value::Packed(weight)),
        (scale_id, Value::Packed(scale)),
    ]);
    drop(owner);
    assert!(weak.upgrade().is_some(), "bindings retain the owner");
    let Value::Host(output) = eval(&graph, &bindings, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| r.output)
        .unwrap()
    else {
        panic!("packed dequant returns the bounded dense oracle");
    };
    assert_eq!(output.shape(), vec![2, 129]);
    drop(bindings);
    assert!(weak.upgrade().is_none(), "last binding releases the owner");
}

#[test]
fn packed_dequant_fused_and_unfused_compositions_match() -> Result<(), BuilderAppendError> {
    for activation_shape in [vec![2, 35], vec![1, 2, 35]] {
        let owner = payload(WeightFormat::E2m1Row32, [3, 35]);
        let descriptor = owner.weight();
        let b = Builder::new();
        let activation = b.slot_named(
            Slot::Activation,
            "packed-test",
            TensorType::f32(activation_shape.clone()),
        );
        let output = packed_linear(&b, activation, "layer", descriptor, None, None)?;
        let graph = b.finish(output);
        let fused = prepare_packed_dequant_production(&graph).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(
            activation.id,
            Value::Host(HostTensor::f32(
                activation_shape.clone(),
                (0..activation_shape.iter().product())
                    .map(|index| (index as f32 - 20.0) / 13.0)
                    .collect(),
            )),
        );
        for graph_input in &graph.inputs {
            let Some(source) = graph
                .meta(*graph_input)
                .name
                .as_deref()
                .and_then(PackedSourceName::parse)
            else {
                continue;
            };
            assert_eq!(source.linear_id(), "layer");
            inputs.insert(
                *graph_input,
                PackedComponentRef::new(Arc::clone(&owner), source.role()).into(),
            );
        }
        let Value::Host(unfused) = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| r.output)
            .unwrap()
        else {
            panic!("dense result");
        };
        let Value::Host(fused) = eval(&fused, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| r.output)
            .unwrap()
        else {
            panic!("dense result");
        };
        assert_eq!(unfused.shape(), fused.shape());
        for (left, right) in unfused
            .as_f32()
            .unwrap()
            .iter()
            .zip(fused.as_f32().unwrap().iter())
        {
            assert!((left - right).abs() <= 1e-5, "{left} != {right}");
        }
    }
    Ok(())
}

/// Mutant M8 (mutants-m4.md): `packed_grouped_linear`'s stable-sort comparators (`ge * -1 + 1` for
/// "selector less" and "row less") were checked only structurally. Five rows routed to experts
/// `[2, 0, 2, 1, 0]` (a tie in expert 0 and in expert 2, so both comparators decide) through three
/// experts with distinct weights: the grouped graph and the indexed graph must both give, row by
/// row, the row's own expert's `x @ W^T` computed from `PackedPayload::decode_row`.
#[test]
fn packed_grouped_linear_matches_the_per_row_expert_product() -> Result<(), BuilderAppendError> {
    use poot_graph_ir::ops::{PackedLinearGraphRow, packed_grouped_linear, packed_indexed_linear};

    let format = WeightFormat::E4m3Block128 {
        scale: ScaleEncoding::E8m0,
    };
    let [out, k] = [3, 5];
    let descriptor = PackedWeight::try_new(format, [out, k]).unwrap();
    let experts: Vec<Arc<PackedPayload>> = (0..3u8)
        .map(|expert| {
            let codes: Vec<u8> = (0..(out * k) as u8)
                .map(|index| (index * 7 + expert * 3 + 1) % 0x70)
                .collect();
            Arc::new(
                PackedPayload::try_new(
                    descriptor,
                    [
                        (SourceRole::Planar(OperandRole::Codes), Arc::from(codes)),
                        (
                            SourceRole::Planar(OperandRole::Scale),
                            Arc::from([0x7f + expert].as_slice()),
                        ),
                    ],
                )
                .unwrap(),
            )
        })
        .collect();
    let rows: Vec<PackedLinearGraphRow> = (0..experts.len())
        .map(|ordinal| PackedLinearGraphRow {
            ordinal,
            linear_id: format!("moe.expert.{ordinal}"),
            descriptor,
        })
        .collect();
    let route = [2usize, 0, 2, 1, 0];
    let m = route.len();
    let x_data: Vec<f32> = (0..m * k)
        .map(|index| (index as f32 - 11.0) / 7.0)
        .collect();

    // Each expert's `[out, K]` weight, row by row through `PackedPayload::decode_row`.
    let weights: Vec<Vec<f32>> = experts
        .iter()
        .map(|expert| {
            let mut weight = vec![0.0f32; out * k];
            for (o, row) in weight.chunks_exact_mut(k).enumerate() {
                expert.decode_row(o, row).unwrap();
            }
            weight
        })
        .collect();
    let mut expected = vec![0.0f32; m * out];
    for (row, &expert) in route.iter().enumerate() {
        for o in 0..out {
            expected[row * out + o] = (0..k)
                .map(|column| x_data[row * k + column] * weights[expert][o * k + column])
                .sum();
        }
    }

    type Build =
        fn(&Builder, Traced, Traced, &[PackedLinearGraphRow]) -> Result<Traced, BuilderAppendError>;
    let builders: [(&str, Build); 2] = [
        ("packed_grouped_linear", packed_grouped_linear),
        ("packed_indexed_linear", packed_indexed_linear),
    ];
    for (name, build) in builders {
        let b = Builder::new();
        let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![m, k]));
        let selector = b.slot_named(Slot::Activation, "selector", TensorType::f32(vec![m]));
        let output = build(&b, x, selector, &rows)?;
        let graph = b.finish(output);
        let mut inputs = HashMap::new();
        inputs.insert(
            x.id,
            Value::Host(HostTensor::f32(vec![m, k], x_data.clone())),
        );
        inputs.insert(
            selector.id,
            Value::Host(HostTensor::f32(
                vec![m],
                route.iter().map(|&expert| expert as f32).collect(),
            )),
        );
        for graph_input in &graph.inputs {
            let Some(source) = graph
                .meta(*graph_input)
                .name
                .as_deref()
                .and_then(PackedSourceName::parse)
            else {
                continue;
            };
            let expert = rows
                .iter()
                .position(|row| row.linear_id == source.linear_id())
                .expect("every packed source names one expert row");
            inputs.insert(
                *graph_input,
                PackedComponentRef::new(Arc::clone(&experts[expert]), source.role()).into(),
            );
        }
        let Value::Host(got) = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| r.output)
            .unwrap()
        else {
            panic!("{name}: dense result");
        };
        assert_eq!(got.shape(), vec![m, out], "{name}");
        for (index, (&got, &want)) in got.as_f32().unwrap().iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                "{name} row {} output {}: {got} vs {want}",
                index / out,
                index % out
            );
        }
    }
    Ok(())
}

/// R467-009: the reversed `(W @ x^T)^T` spelling of `packed_linear`'s canonical `x @ W^T`, recognized as
/// the same `PackedContraction` and matched against the unfused reversed composition through the
/// generic dense-equation path, exactly like the sibling above. Built through the same public
/// append-plan API `ops::packed_linear` itself uses (`Builder::append_plan`/`equation`), never a
/// hand-poked `ValueMeta` (`key` is crate-private outside `poot-graph-ir`).
///
/// The square case (`batch == k`) is the one that actually exercises the numeric comparison rather than
/// the recognizer's shape check: `recognize_transposed_canonical_packed_chain`'s replacement must read
/// the *untransposed* activation as `PackedContraction`'s first operand, not the `x^T` the inner `MatMul`
/// reads. For a non-square activation, reading `x^T` there disagrees in shape with the declared output
/// and the shared infer-and-compare gate declines it (a loud failure). At `batch == k`, `x` and `x^T`
/// have the *same* shape, so that bug would still recognize and shape-check - only a numeric oracle
/// comparison catches it, the same class of bug `blocked_packed_contraction_fused_and_unfused_compositions_match`'s
/// own comment names. Runs the square case first so a reversed-operand mutation is caught here, not
/// merely declined by the second (non-square) case before this assertion runs.
#[test]
fn transposed_packed_contraction_fused_and_unfused_compositions_match()
-> Result<(), BuilderAppendError> {
    for activation_shape in [vec![4usize, 4], vec![2, 4]] {
        let owner = payload(WeightFormat::E2m1Row32, [3, 4]);
        let descriptor = owner.weight();
        let b = Builder::new();
        let activation = b.slot_named(
            Slot::Activation,
            "transposed-packed-test",
            TensorType::f32(activation_shape.clone()),
        );

        let mut plan = b.append_plan(0);
        let [(weight_name, weight_type), (scale_name, scale_type)] =
            <[(PackedSourceName, TensorType); 2]>::try_from(packed_source_constants(
                "transposed-packed-test",
                descriptor,
            ))
            .expect("a registered packed-linear format has exactly two sources");
        let weight_source = plan.input(weight_name, weight_type, poot_graph_ir::Storage::Const)?;
        let scale_source = plan.input(scale_name, scale_type, poot_graph_ir::Storage::Const)?;
        let weight = plan.equation(
            OpKind::PackedDequant { descriptor },
            vec![
                Operand::Value(weight_source.id),
                Operand::Value(scale_source.id),
            ],
        )?;
        let activation_t = plan.equation(
            OpKind::Transpose { perm: vec![1, 0] },
            vec![Operand::Value(activation.id)],
        )?;
        let product = plan.equation(
            OpKind::MatMul,
            vec![Operand::Value(weight.id), Operand::Value(activation_t.id)],
        )?;
        let y = plan.equation(
            OpKind::Transpose { perm: vec![1, 0] },
            vec![Operand::Value(product.id)],
        )?;
        plan.declare_result(y)?;
        let mut prepared = b.preflight_append(plan)?;
        let id = b.commit_append(&mut prepared)?;
        let graph = b.finish(Traced { id });

        let fused = prepare_packed_dequant_production(&graph).unwrap();
        assert!(
            matches!(
                fused.eqns.as_slice(),
                [poot_graph_ir::Eqn {
                    op: OpKind::PackedContraction { blocks: 1, .. },
                    ..
                }]
            ),
            "expected one PackedContraction: {fused:?}"
        );

        let mut inputs = HashMap::new();
        inputs.insert(
            activation.id,
            Value::Host(HostTensor::f32(
                activation_shape.clone(),
                (0..activation_shape.iter().product())
                    .map(|index| (index as f32 - 20.0) / 13.0)
                    .collect(),
            )),
        );
        for graph_input in &graph.inputs {
            let Some(source) = graph
                .meta(*graph_input)
                .name
                .as_deref()
                .and_then(PackedSourceName::parse)
            else {
                continue;
            };
            assert_eq!(source.linear_id(), "transposed-packed-test");
            inputs.insert(
                *graph_input,
                PackedComponentRef::new(Arc::clone(&owner), source.role()).into(),
            );
        }
        let Value::Host(unfused) = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| r.output)
            .unwrap()
        else {
            panic!("dense result");
        };
        let Value::Host(fused_output) =
            eval(&fused, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
                .map(|r| r.output)
                .unwrap()
        else {
            panic!("dense result");
        };
        assert_eq!(unfused.shape(), fused_output.shape());
        for (left, right) in unfused
            .as_f32()
            .unwrap()
            .iter()
            .zip(fused_output.as_f32().unwrap().iter())
        {
            assert!(
                left.is_finite() && right.is_finite(),
                "non-finite element: {left} vs {right}"
            );
            assert!((left - right).abs() <= 1e-5, "{left} != {right}");
        }
    }
    Ok(())
}

/// Card 385: `ops::packed_block_diagonal_linear`'s fused `PackedContraction { blocks, .. }` must match its unfused
/// `PackedDequant -> Reshape -> Transpose([0,2,1]) -> MatMul` composition, evaluated through the generic dense-equation
/// path (not a second hand-written reference). A wrong axis order in the fused path (e.g. reading
/// `weight[j, block_base + output]` instead of `weight[block_base + output, j]`) diverges from the unfused result.
#[test]
fn blocked_packed_contraction_fused_and_unfused_compositions_match()
-> Result<(), BuilderAppendError> {
    for activation_shape in [vec![4, 5, 35], vec![4, 1, 35]] {
        let owner = payload(WeightFormat::E2m1Row32, [8, 35]);
        let descriptor = owner.weight();
        let b = Builder::new();
        let activation = b.slot_named(
            Slot::Activation,
            "blocked-packed-test",
            TensorType::f32(activation_shape.clone()),
        );
        let output =
            packed_block_diagonal_linear(&b, activation, "blocked-layer", descriptor, 4, None)?;
        let graph = b.finish(output);
        let fused = prepare_packed_dequant_production(&graph).unwrap();
        assert!(matches!(
            fused.eqns.as_slice(),
            [poot_graph_ir::Eqn {
                op: poot_graph_ir::OpKind::PackedContraction { blocks: 4, .. },
                ..
            }]
        ));
        let mut inputs = HashMap::new();
        inputs.insert(
            activation.id,
            Value::Host(HostTensor::f32(
                activation_shape.clone(),
                (0..activation_shape.iter().product())
                    .map(|index| (index as f32 - 20.0) / 13.0)
                    .collect(),
            )),
        );
        for graph_input in &graph.inputs {
            let Some(source) = graph
                .meta(*graph_input)
                .name
                .as_deref()
                .and_then(PackedSourceName::parse)
            else {
                continue;
            };
            assert_eq!(source.linear_id(), "blocked-layer");
            inputs.insert(
                *graph_input,
                PackedComponentRef::new(Arc::clone(&owner), source.role()).into(),
            );
        }
        let Value::Host(unfused) = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| r.output)
            .unwrap()
        else {
            panic!("dense result");
        };
        let Value::Host(fused_output) =
            eval(&fused, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
                .map(|r| r.output)
                .unwrap()
        else {
            panic!("dense result");
        };
        assert_eq!(unfused.shape(), fused_output.shape());
        for (left, right) in unfused
            .as_f32()
            .unwrap()
            .iter()
            .zip(fused_output.as_f32().unwrap().iter())
        {
            assert!((left - right).abs() <= 1e-5, "{left} != {right}");
        }
    }
    Ok(())
}

#[test]
fn packed_dequant_production_rejects_valid_preexisting_candidate() -> Result<(), BuilderAppendError>
{
    let owner = payload(WeightFormat::E2m1Row32, [3, 35]);
    let descriptor = owner.weight();
    let builder = Builder::new();
    let activation = builder.slot_named(
        Slot::Activation,
        "direct-candidate",
        TensorType::f32(vec![2, 35]),
    );
    let output = packed_linear(&builder, activation, "direct.layer", descriptor, None, None)?;
    let source = builder.finish(output);
    let direct = prepare_packed_dequant_production(&source).unwrap();
    assert!(matches!(
        direct.eqns.as_slice(),
        [poot_graph_ir::Eqn {
            op: poot_graph_ir::OpKind::PackedContraction {
                descriptor: actual,
                blocks: 1
            },
            ..
        }] if *actual == descriptor
    ));

    let candidate = &direct.eqns[0];
    let [
        _,
        poot_graph_ir::Operand::Value(weight_id),
        poot_graph_ir::Operand::Value(scale_id),
    ] = candidate.inputs.as_slice()
    else {
        panic!("candidate must retain activation, weight, and scale values");
    };
    let weight =
        PackedComponentRef::new(Arc::clone(&owner), SourceRole::Planar(OperandRole::Codes));
    let scale = PackedComponentRef::new(Arc::clone(&owner), SourceRole::Planar(OperandRole::Scale));
    assert!(weight.same_owner(&scale));
    let inputs = HashMap::from([
        (
            activation.id,
            Value::Host(HostTensor::f32(
                vec![2, 35],
                (0..70).map(|index| index as f32 / 17.0).collect(),
            )),
        ),
        (*weight_id, Value::Packed(weight)),
        (*scale_id, Value::Packed(scale)),
    ]);
    assert!(matches!(
        eval(&direct, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).map(|r| r.output),
        Ok(Value::Host(_))
    ));
    assert!(matches!(
        prepare_packed_dequant_production(&direct),
        Err(PackedDequantProductionError::PreexistingContraction { .. })
    ));
    Ok(())
}

/// Evaluate one `OpKind::PackedContraction { blocks: 1 }` of `activation` (`[m, K]`) against
/// `owner` through the graph oracle: `[m, out]`, the activation times the packed weight's
/// transpose.
fn contract_through_the_oracle(
    label: &str,
    owner: &Arc<PackedPayload>,
    activation: HostTensor,
) -> Result<HostTensor, BuilderAppendError> {
    let weight = owner.weight();
    let b = Builder::new();
    let activation_value = b.slot_named(
        Slot::Activation,
        label,
        TensorType::f32(activation.shape().to_vec()),
    );
    let mut plan = b.append_plan(0);
    let mut operands = vec![Operand::Value(activation_value.id)];
    let mut source_ids = Vec::new();
    for (name, ty) in packed_source_constants(label, weight) {
        let role = name.role();
        let input = plan.input(name, ty, poot_graph_ir::Storage::Const)?;
        source_ids.push((input.id, role));
        operands.push(Operand::Value(input.id));
    }
    let output = plan.equation(
        OpKind::PackedContraction {
            descriptor: weight,
            blocks: 1,
        },
        operands,
    )?;
    plan.declare_result(output)?;
    let mut prepared = b.preflight_append(plan)?;
    let id = b.commit_append(&mut prepared)?;
    let graph = b.finish(Traced { id });

    let mut inputs = HashMap::new();
    inputs.insert(activation_value.id, Value::Host(activation));
    for (source_id, role) in &source_ids {
        inputs.insert(
            *source_id,
            Value::Packed(PackedComponentRef::new(Arc::clone(owner), *role)),
        );
    }
    let Value::Host(output) = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| r.output)
        .unwrap_or_else(|error| panic!("{label}: eval failed: {error}"))
    else {
        panic!("{label}: packed contraction must produce dense oracle output");
    };
    Ok(output)
}

/// Mutant H0 (mutants-m4.md), at the oracle's one packed read: `PackedPayload::decode_row`'s
/// block-branch row offset (`crates/poot-quant/src/lib.rs:638`, `let start = row * row_bytes;`,
/// reached through `decode_packed_row`) made `/` reads row 0 for every row of a block-format
/// weight, and every block case of the 539a literal test is one row, so nothing failed. The same
/// line is killed in poot-quant by `block_payload_decode_reads_the_addressed_row`. A multi-row Q8_0 and Q4_K
/// weight with a distinct f16 `d` per row (ggml `block_q8_0` and `block_q4_K` both start with it;
/// `block_q4_K`'s `dmin` follows), contracted with an identity activation, must give each row's
/// own `decode_blocks` values.
#[test]
fn packed_dequant_oracle_matches_decode_blocks_per_row() -> Result<(), BuilderAppendError> {
    use poot_quant::format::Storage;

    const D: [u16; 3] = [0x3c00, 0x4000, 0xb800];
    for (format, [out, k]) in [
        (WeightFormat::Q8_0, [3, 64]),
        (WeightFormat::Q4_K, [3, 256]),
    ] {
        let descriptor = format.descriptor();
        let Storage::Blocks(layout) = descriptor.storage else {
            unreachable!("block formats only")
        };
        let row_bytes = k / layout.values * layout.bytes;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut bytes: Vec<u8> = (0..out * row_bytes)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect();
        for (index, block) in bytes.chunks_exact_mut(layout.bytes).enumerate() {
            block[0..2].copy_from_slice(&D[index * layout.bytes / row_bytes].to_le_bytes());
            if format == WeightFormat::Q4_K {
                block[2..4].copy_from_slice(&0x3400u16.to_le_bytes());
            }
        }
        let weight = PackedWeight::try_new(format, [out, k]).unwrap();
        let owner = Arc::new(
            PackedPayload::try_new(weight, [(SourceRole::Blocks, Arc::from(bytes.as_slice()))])
                .unwrap(),
        );

        let mut identity = vec![0.0f32; k * k];
        for column in 0..k {
            identity[column * k + column] = 1.0;
        }
        let label = format!("{format:?}");
        let output =
            contract_through_the_oracle(&label, &owner, HostTensor::f32(vec![k, k], identity))?;
        assert_eq!(output.shape(), vec![k, out]);

        let mut rows = Vec::new();
        for (row, row_source) in bytes.chunks_exact(row_bytes).enumerate() {
            let mut want = vec![0.0f32; k];
            descriptor.decode_blocks(row_source, &mut want).unwrap();
            for (column, &want) in want.iter().enumerate() {
                let got = output.as_f32().unwrap()[column * out + row];
                assert!(
                    got == want,
                    "{format:?} [{row}, {column}]: oracle {got:?}, decode_blocks {want:?}"
                );
            }
            rows.push(want);
        }
        assert!(
            rows[0] != rows[1] && rows[1] != rows[2] && rows[0] != rows[2],
            "{format:?}: the fixture rows must decode differently"
        );
    }
    Ok(())
}

/// SC-001: for every scheme the IR admits - the 13 GGUF block formats, the 6 E4M3 per-channel/128x128
/// scale variants, E2M1 row-32, and GPTQ (contiguous and act-order) and AWQ - the `PackedContraction`
/// oracle's matmul against a one-hot activation equals Card 539a's llama.cpp/OCP/AutoGPTQ/AutoAWQ-cited
/// literal reference value at that activation's own K index, bit for bit (ADR-0101 tier 1). Every
/// literal hex block/tensor and its `(index, value)` (or `([out,k], value)`) pairs is copied verbatim
/// from `poot_quant::blocks`'s and `poot_quant::planar`'s own `#[cfg(test)]` reference vectors, not
/// re-derived: the expected values are 539a's literals, never a second call of the decoder.
///
/// Mutation (run 2026-09-29): flipped the sign of `poot_quant::blocks::Q4_0`'s code offset
/// (`FieldEncoding::Unsigned { offset: -8 }` -> `{ offset: 8 }`) ->
/// `packed_dequant_oracle_matches_the_539a_literal_reference_for_every_scheme` failed at `Q4_0 [0,0]:
/// got 0.20553589, expected (dense matmul over literal -0.013702393) -0.013702393` (`left:
/// 1045592064, right: 3160440832`). Restored.
#[test]
fn packed_dequant_oracle_matches_the_539a_literal_reference_for_every_scheme()
-> Result<(), BuilderAppendError> {
    use poot_quant::format::{GroupMap, ScaleEncoding, WeightFormat};

    fn nonzero(value: usize) -> std::num::NonZeroUsize {
        std::num::NonZeroUsize::new(value).unwrap()
    }

    fn hex(text: &str) -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
            .collect()
    }

    /// `poot_quant::planar`'s `assert_e4m3` test helper's weight formula: byte `(o*131 + k*7 + 3) %
    /// 256`, with the two NaN E4M3 codes (`0x7f`, `0xff`) replaced by zero, exactly as that helper
    /// builds its own fixture.
    fn e4m3_weight_bytes(shape: [usize; 2]) -> Vec<u8> {
        (0..shape[0])
            .flat_map(|o| (0..shape[1]).map(move |k| (o * 131 + k * 7 + 3) % 256))
            .map(|byte| match byte as u8 {
                0x7f | 0xff => 0,
                byte => byte,
            })
            .collect()
    }

    struct Case {
        label: &'static str,
        format: WeightFormat,
        shape: [usize; 2],
        sources: Vec<(SourceRole, Vec<u8>)>,
        /// `([out, k], expected)`.
        points: Vec<([usize; 2], f32)>,
    }

    fn block_case(
        label: &'static str,
        format: WeightFormat,
        values: usize,
        block_hex: &str,
        points: &[(usize, f32)],
    ) -> Case {
        Case {
            label,
            format,
            shape: [1, values],
            sources: vec![(SourceRole::Blocks, hex(block_hex))],
            points: points.iter().map(|&(k, v)| ([0, k], v)).collect(),
        }
    }

    let block_cases = [
        block_case(
            "Q4_0",
            WeightFormat::Q4_0,
            32,
            "042397621649bd7821226d9d3dd30a06d54a",
            &[
                (0, -0.013702393),
                (3, 0.013702393),
                (6, -0.09591675),
                (9, 0.06851196),
                (12, 0.027404785),
                (15, 0.027404785),
                (18, -0.09591675),
                (21, -0.013702393),
                (24, -0.027404785),
                (27, 0.06851196),
                (30, 0.06851196),
                (31, -0.05480957),
            ],
        ),
        block_case(
            "Q4_1",
            WeightFormat::Q4_1,
            32,
            "0423b8b21d145901fadc73a8ffbf47d8a2a2457b",
            &[
                (0, -0.031829834),
                (3, -0.19625854),
                (6, -0.16885376),
                (9, -0.004425049),
                (12, -0.18255615),
                (15, -0.05923462),
                (18, -0.14144897),
                (21, -0.031829834),
                (24, -0.004425049),
                (27, -0.031829834),
                (30, -0.15515137),
                (31, -0.11404419),
            ],
        ),
        block_case(
            "Q5_0",
            WeightFormat::Q5_0,
            32,
            "0423ee8e46327f2c16fa536988e9d93c837043d73aeb",
            &[
                (0, -0.013702393),
                (3, 0.13702393),
                (6, 0.10961914),
                (9, 0.16442871),
                (12, -0.1781311),
                (15, 0.15072632),
                (18, 0.013702393),
                (21, -0.13702393),
                (24, -0.041107178),
                (27, -0.12332153),
                (30, -0.1781311),
                (31, -0.027404785),
            ],
        ),
        block_case(
            "Q5_1",
            WeightFormat::Q5_1,
            32,
            "0423b8b2490a0e50da8a62ee274c95b68d565f37472ea84e",
            &[
                (0, 0.14630127),
                (3, 0.20111084),
                (6, 0.07778931),
                (9, 0.0914917),
                (12, -0.11404419),
                (15, -0.018127441),
                (18, 0.0914917),
                (21, -0.15515137),
                (24, -0.1003418),
                (27, -0.16885376),
                (30, 0.14630127),
                (31, -0.15515137),
            ],
        ),
        block_case(
            "Q8_0",
            WeightFormat::Q8_0,
            32,
            "04230f9e167c0081016eefe8f408aad58096763bee663bf085c2c81e773207646c57",
            &[
                (0, 0.20553589),
                (3, 1.6990967),
                (6, 0.013702393),
                (9, -0.32885742),
                (12, -1.1784058),
                (15, -1.4524536),
                (18, -0.24664307),
                (21, -0.21923828),
                (24, -0.767334),
                (27, 0.6851196),
                (30, 1.4798584),
                (31, 1.1921082),
            ],
        ),
        block_case(
            "Q2_K",
            WeightFormat::Q2_K,
            256,
            concat!(
                "12ff11bdd71c566ee39a03f9ebde5f9e3eeafd5ab4f4aa3511799ed220538bc8",
                "80deddf9d3d7baf93f42bdc17e7977f42729f4435ff7baca9e8328d9fa4e3184",
                "0698d2fd7862a239e04cf9ef504b2044451fc921",
            ),
            &[
                (0, 0.017097473),
                (11, 0.017097473),
                (22, 0.04348755),
                (33, 0.00289917),
                (44, -0.011299133),
                (55, 0.060287476),
                (66, 0.002193451),
                (77, -0.09719467),
                (88, 0.24427032),
                (99, -0.013900757),
                (110, 0.028694153),
                (121, 0.031593323),
                (132, -0.0942955),
                (143, -0.15818787),
                (154, -0.030700684),
                (165, 0.021297455),
                (176, -0.105594635),
                (187, 0.022190094),
                (198, 0.07608414),
                (209, -0.04750061),
                (220, -0.04750061),
                (231, 0.26296616),
                (242, 0.19647217),
                (253, -0.0023040771),
                (255, -0.0023040771),
            ],
        ),
        block_case(
            "Q3_K",
            WeightFormat::Q3_K,
            256,
            concat!(
                "513cd71855113ac60e98a370705c0c6cc5a6d6b9e91d0b38a023e0ae014fbda0",
                "4709e8fffcf2cff8d6c7c057de7e9ff123903238bbe563f56ec7b266bff9c474",
                "76287bad196dbc8999aa926c0937bcfeaf0036aa9d9d1ce704c9815872eabf63",
                "c17e9754da3ea7405f12bf6a451f",
            ),
            &[
                (0, 0.36205673),
                (11, -0.12068558),
                (22, 0.29816437),
                (33, -0.32656097),
                (44, -0.16328049),
                (55, -0.08518982),
                (66, 0.36915588),
                (77, 0.5537338),
                (88, 0.25556946),
                (99, 0.48984146),
                (110, 0.32656097),
                (121, -0.0),
                (132, -0.028396606),
                (143, 0.056793213),
                (154, 0.1916771),
                (165, -0.17747879),
                (176, -0.035495758),
                (187, 0.070991516),
                (198, 0.021297455),
                (209, 0.8235016),
                (220, 0.2058754),
                (231, 0.14198303),
                (242, -0.0),
                (253, 0.08518982),
                (255, -0.08518982),
            ],
        ),
        block_case(
            "Q4_K",
            WeightFormat::Q4_K,
            256,
            concat!(
                "451fc9219d1b645efb1d0f891f7c9a3575baad5711b8b0937499e433c92297b3",
                "75d815440095827c5450bb1709de3034def600c254b5c9d329f373a8c85aba7e",
                "e354b9a43dae4cf652843c4f9a82d24f47705a7fd09054d8eb19d3fc8eee07ed",
                "1bce13c0b61a602e2962b5d55aa5f7a312de08a01e726c42af359e0fc5d702ae",
                "dbd3a000d371cb5ca84a616b0201a2ac",
            ),
            &[
                (0, 0.36272812),
                (11, -0.049022675),
                (22, -0.25489807),
                (33, 1.7807732),
                (44, 1.9724503),
                (55, 1.0140648),
                (66, -0.169487),
                (77, 2.3862076),
                (88, 0.34165192),
                (99, 2.4540024),
                (110, 2.2410278),
                (121, 1.6021042),
                (132, -0.55365753),
                (143, 3.783924),
                (154, 1.1146431),
                (165, 0.68761444),
                (176, 0.006095886),
                (187, 1.0283737),
                (198, 2.113243),
                (209, 0.45204163),
                (220, 0.26746368),
                (231, 0.20085907),
                (242, 1.0953522),
                (253, -0.39546967),
                (255, 1.0953522),
            ],
        ),
        block_case(
            "Q5_K",
            WeightFormat::Q5_K,
            256,
            concat!(
                "451fc921c0478de684332890a3cc170b9b6de18c07870e9fad5242c340f1d5f3",
                "571a647f5d2580560c84c9a3c6128d5cb3f309e886381fda2c4f2e5f419e6899",
                "988df40b2002fc09c4d8045f8a98e7cbaf877ca5af50ed49c394b683b8add5de",
                "aaa0e4a1690aef046454c8380c066eb53f0e94b260965b82c75beb9f01c732f2",
                "63394909d8fb84c14094526f8fed8465b7b9bad61847131139407708ebbfa851",
                "469289e2813e364cc0e13624b2ea49ec",
            ),
            &[
                (0, -0.045196533),
                (11, -0.045196533),
                (22, -0.045196533),
                (33, 0.16915512),
                (44, -0.37747955),
                (55, 0.21884918),
                (66, 0.6555023),
                (77, 0.7477913),
                (88, 1.3938141),
                (99, 6.8331757),
                (110, 3.3261948),
                (121, 1.1680527),
                (132, -0.4745636),
                (143, 6.0424576),
                (154, 0.24954987),
                (165, 1.6533966),
                (176, 1.0570679),
                (187, 4.237488),
                (198, 0.8193016),
                (209, 0.5424347),
                (220, 4.9723053),
                (231, 6.758877),
                (242, 2.9892273),
                (253, 5.502327),
                (255, 5.502327),
            ],
        ),
        block_case(
            "Q6_K",
            WeightFormat::Q6_K,
            256,
            concat!(
                "4293ba0b32a291d6db143d29f323d53c314f8968cf76ea446fa012c169ad04a3",
                "b59907cf091577286170d493bec0a383ebeac662b4b4425ea26c6c3e7dac9d55",
                "137c0107d87f14ec47749c4ed8f12633cb5508cd1f648aacc09e3857cb28a01e",
                "8a4758942e6d4a7412b3d5e0d295debababe2a918bf1468a85ea8249d7e8ff23",
                "54dd2c0b4c4883524b6155bdd59bad0b037011bec8b1dca798afd5061caae997",
                "29cb881d7ff762bd76afc455feca077095647b733ccbb555ec8a3f9a98d0ef75",
                "6013767cf76acf8d4403a28f52346e83451f",
            ),
            &[
                (0, -20.445557),
                (11, -4.77063),
                (22, -2.9674454),
                (33, 20.942497),
                (44, -1.6753998),
                (55, -1.7605896),
                (66, -0.702816),
                (77, 0.8944931),
                (88, -7.5251007),
                (99, 6.9571686),
                (110, -3.4785843),
                (121, -4.8984146),
                (132, 11.585815),
                (143, -13.999527),
                (154, 0.5111389),
                (165, 2.0019608),
                (176, 4.813225),
                (187, -7.219837),
                (198, 0.58213043),
                (209, 1.8457794),
                (220, -1.4766235),
                (231, 5.4663467),
                (242, 12.423515),
                (253, -26.621819),
                (255, 12.423515),
            ],
        ),
        block_case(
            "Iq4_Nl",
            WeightFormat::Iq4_Nl,
            32,
            "04238760260d2c74c00575dfcf611de25bd8",
            &[
                (0, -0.13702393),
                (3, 0.9454651),
                (6, -1.7402039),
                (9, 1.5483704),
                (12, 0.9454651),
                (15, 0.013702393),
                (18, -1.1372986),
                (21, -0.13702393),
                (24, -0.13702393),
                (27, -0.30145264),
                (30, -0.47958374),
                (31, 0.9454651),
            ],
        ),
        block_case(
            "Iq4_Xs",
            WeightFormat::Iq4_Xs,
            256,
            concat!(
                "451f40ea9cb8b3a47f4a630b23f32b3622e927be470ce34ff63d9ec4db621023",
                "dfce18a8378c574185433b3128c79dbbefe90a0690397faf39fb09b28497e655",
                "6cf7c412c93121c3192ba819812b1786651383708c1e4f244e9ece2edc660fb4",
                "a25109ca0cf962c13e0018aead9babf9d1e4f9ac07d10b342fa353897ce27c29",
                "1357df59f780d916",
            ),
            &[
                (0, -16.044083),
                (11, -12.63649),
                (22, 11.784592),
                (33, -11.266354),
                (44, 1.6328049),
                (55, 13.55228),
                (66, -6.4744263),
                (77, -2.2149353),
                (88, -15.163788),
                (99, 2.946148),
                (110, 3.6915588),
                (121, -4.0110207),
                (132, -2.2149353),
                (143, -1.0435753),
                (154, 0.021297455),
                (165, -1.7179947),
                (176, -3.8264427),
                (187, 4.1388054),
                (198, 1.079071),
                (209, -3.606369),
                (220, -3.606369),
                (231, 2.3995132),
                (242, -6.460228),
                (253, 0.18457794),
                (255, -19.196106),
            ],
        ),
        block_case(
            "Mxfp4",
            WeightFormat::Mxfp4,
            32,
            "7c5c44b968cff4eb51b989d24e70cf4430",
            &[
                (0, -0.25),
                (3, -0.0),
                (6, -0.1875),
                (9, -0.0625),
                (12, 0.0),
                (15, 0.0),
                (18, -0.1875),
                (21, -0.75),
                (24, -0.1875),
                (27, 0.25),
                (30, 0.25),
                (31, 0.1875),
            ],
        ),
    ];

    let e4m3_per_channel_shape = [3usize, 5];
    let e4m3_block128_shape = [129usize, 130];
    let e4m3_weight_per_channel = e4m3_weight_bytes(e4m3_per_channel_shape);
    let e4m3_weight_block128 = e4m3_weight_bytes(e4m3_block128_shape);

    let planar_cases = vec![
        Case {
            label: "E4m3PerChannel/F32",
            format: WeightFormat::E4m3PerChannel {
                scale: ScaleEncoding::F32,
            },
            shape: e4m3_per_channel_shape,
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    e4m3_weight_per_channel.clone(),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    hex("0000003f0000e0bfa69b443b"),
                ),
            ],
            points: vec![
                ([0, 0], 0.0029296875),
                ([0, 1], 0.009765625),
                ([0, 2], 0.017578125),
                ([0, 3], 0.03125),
                ([0, 4], 0.05859375),
                ([1, 0], 0.020507812),
                ([1, 1], 0.044433594),
                ([1, 2], 0.08203125),
                ([1, 3], 0.15039062),
                ([1, 4], 0.2734375),
                ([2, 0], 5.2734376e-5),
                ([2, 1], 9.375e-5),
                ([2, 2], 0.00017578126),
                ([2, 3], 0.000328125),
                ([2, 4], 0.000609375),
            ],
        },
        Case {
            label: "E4m3PerChannel/Bf16",
            format: WeightFormat::E4m3PerChannel {
                scale: ScaleEncoding::Bf16,
            },
            shape: e4m3_per_channel_shape,
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    e4m3_weight_per_channel.clone(),
                ),
                (SourceRole::Planar(OperandRole::Scale), hex("403f80bf003c")),
            ],
            points: vec![
                ([0, 0], 0.0043945312),
                ([0, 1], 0.0146484375),
                ([0, 2], 0.026367188),
                ([0, 3], 0.046875),
                ([0, 4], 0.087890625),
                ([1, 0], 0.01171875),
                ([1, 1], 0.025390625),
                ([1, 2], 0.046875),
                ([1, 3], 0.0859375),
                ([1, 4], 0.15625),
                ([2, 0], 0.0001373291),
                ([2, 1], 0.00024414062),
                ([2, 2], 0.00045776367),
                ([2, 3], 0.0008544922),
                ([2, 4], 0.0015869141),
            ],
        },
        Case {
            label: "E4m3PerChannel/E8m0",
            format: WeightFormat::E4m3PerChannel {
                scale: ScaleEncoding::E8m0,
            },
            shape: e4m3_per_channel_shape,
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    e4m3_weight_per_channel.clone(),
                ),
                (SourceRole::Planar(OperandRole::Scale), hex("7e8278")),
            ],
            points: vec![
                ([0, 0], 0.0029296875),
                ([0, 1], 0.009765625),
                ([0, 2], 0.017578125),
                ([0, 3], 0.03125),
                ([0, 4], 0.05859375),
                ([1, 0], -0.09375),
                ([1, 1], -0.203125),
                ([1, 2], -0.375),
                ([1, 3], -0.6875),
                ([1, 4], -1.25),
                ([2, 0], 0.0001373291),
                ([2, 1], 0.00024414062),
                ([2, 2], 0.00045776367),
                ([2, 3], 0.0008544922),
                ([2, 4], 0.0015869141),
            ],
        },
        Case {
            label: "E4m3Block128/F32",
            format: WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::F32,
            },
            shape: e4m3_block128_shape,
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    e4m3_weight_block128.clone(),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    // `poot_quant::planar`'s own literal test pins scale index 1 to `-2.0` (negative on
                    // purpose, to prove the scale field's sign math), covered by `[0,128]`, `[0,129]`
                    // and `[64,129]`. `PackedPayload::try_new` additionally enforces the *production*
                    // content invariant that a registered E4M3 cell's scale is finite and positive
                    // (`validate_scale_encodings`), so this test flips that one block's scale to `2.0`
                    // (same magnitude, sign math already pinned by `poot_quant::planar`'s own test) and
                    // drops the three points it decoded, testing the other 8 through the full graph
                    // oracle this test drives.
                    hex("0000803e000000408fc2753c0000e040"),
                ),
            ],
            points: vec![
                ([0, 0], 0.0014648438),
                ([0, 127], 96.0),
                ([127, 0], 0.0),
                ([127, 127], 72.0),
                ([128, 0], -8.789062e-5),
                ([128, 128], 0.041015625),
                ([128, 129], 0.13671875),
                ([5, 77], -0.1015625),
            ],
        },
        Case {
            label: "E4m3Block128/Bf16",
            format: WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::Bf16,
            },
            shape: e4m3_block128_shape,
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    e4m3_weight_block128.clone(),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    // See the F32 case above: same fix, same three points dropped.
                    hex("803e0040403fe040"),
                ),
            ],
            points: vec![
                ([0, 0], 0.0014648438),
                ([0, 127], 96.0),
                ([127, 0], 0.0),
                ([127, 127], 72.0),
                ([128, 0], -0.0043945312),
                ([128, 128], 0.041015625),
                ([128, 129], 0.13671875),
                ([5, 77], -0.1015625),
            ],
        },
        Case {
            label: "E4m3Block128/E8m0",
            format: WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::E8m0,
            },
            shape: e4m3_block128_shape,
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    e4m3_weight_block128.clone(),
                ),
                (SourceRole::Planar(OperandRole::Scale), hex("7d807981")),
            ],
            points: vec![
                ([0, 0], 0.0014648438),
                ([0, 127], 96.0),
                ([0, 128], -0.01171875),
                ([0, 129], -0.0390625),
                ([127, 0], 0.0),
                ([127, 127], 72.0),
                ([128, 0], -9.1552734e-5),
                ([128, 128], 0.0234375),
                ([128, 129], 0.078125),
                ([64, 129], 1.0e1),
                ([5, 77], -0.1015625),
            ],
        },
        Case {
            label: "E2m1Row32",
            format: WeightFormat::E2m1Row32,
            shape: [2, 35],
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    hex("80837d1d0612d0e601c6d1bdf94dad156304c0dca8af7231d39ce31efb211b9916741c0a"),
                ),
                (SourceRole::Planar(OperandRole::Scale), hex("7f817a02")),
            ],
            points: vec![
                ([0, 0], 0.0),
                ([0, 1], -0.0),
                ([0, 2], 1.5),
                ([0, 17], 0.0),
                ([0, 30], 3.0),
                ([0, 31], 0.5),
                ([0, 32], 6.0),
                ([0, 33], 16.0),
                ([0, 34], 8.0),
                ([1, 0], 0.0),
                ([1, 1], -0.0625),
                ([1, 2], -0.0625),
                ([1, 17], -0.125),
                ([1, 30], 0.0625),
                ([1, 31], 0.1875),
                ([1, 32], -4.7019774e-38),
                ([1, 33], 1.1754944e-38),
                ([1, 34], -2.3509887e-38),
            ],
        },
        Case {
            label: "Gptq/Contiguous",
            format: WeightFormat::Gptq {
                groups: GroupMap::Contiguous { size: nonzero(8) },
            },
            shape: [8, 16],
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    hex(
                        "6187f803f0423141182c8d58e2912c82c2b7f4045a572fb18fe594e3b5e79e9806fada64fa70e9ce9a1e4d9d1e4970acab2968c3f084cc5f1de2b0ccc9cd39c1",
                    ),
                ),
                (
                    SourceRole::Planar(OperandRole::Zero),
                    hex("c0ad6caaba893613"),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    hex("aea7c32500310a27ae273daab8aa33abaea7c321cda87e21ae233daa7b287d23"),
                ),
            ],
            points: vec![
                ([0, 0], -0.0),
                ([1, 1], 0.045013428),
                ([7, 7], 0.11248779),
                ([3, 8], 0.053634644),
                ([5, 9], -0.5361023),
                ([7, 15], 0.1462555),
                ([2, 12], -0.11251831),
                ([6, 4], 0.36743164),
            ],
        },
        Case {
            label: "Gptq/Indexed",
            format: WeightFormat::Gptq {
                groups: GroupMap::Indexed { groups: nonzero(2) },
            },
            shape: [8, 16],
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    hex(
                        "6187f803f0423141182c8d58e2912c82c2b7f4045a572fb18fe594e3b5e79e9806fada64fa70e9ce9a1e4d9d1e4970acab2968c3f084cc5f1de2b0ccc9cd39c1",
                    ),
                ),
                (
                    SourceRole::Planar(OperandRole::Zero),
                    hex("c0ad6caaba893613"),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    hex("aea7c32500310a27ae273daab8aa33abaea7c321cda87e21ae233daa7b287d23"),
                ),
                (
                    SourceRole::Planar(OperandRole::GroupIndex),
                    hex(
                        "01000000000000000100000000000000010000000000000001000000000000000100000000000000010000000000000001000000000000000100000000000000",
                    ),
                ),
            ],
            points: vec![
                ([0, 0], 0.2999878),
                ([1, 1], 0.045013428),
                ([7, 7], 0.11248779),
                ([3, 8], 0.053634644),
                ([5, 9], -0.38989258),
                ([7, 15], -0.056243896),
                ([2, 12], -0.11251831),
                ([6, 4], 0.0),
            ],
        },
        Case {
            label: "Awq",
            format: WeightFormat::Awq {
                group_size: nonzero(8),
            },
            shape: [8, 16],
            sources: vec![
                (
                    SourceRole::Planar(OperandRole::Codes),
                    hex(
                        "81f2205a168cefb5c757127728eb94e5d844c1ef8f9f23928334218150e0849ba6dbea90901a1fcfea2990d41fe247c8da08099c4db67e3cd4c3ce1f96ccacc5",
                    ),
                ),
                (
                    SourceRole::Planar(OperandRole::Zero),
                    hex("d0acaca69a368b13"),
                ),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    hex("aea7c32500310a27ae273daab8aa33abaea7c321cda87e21ae233daa7b287d23"),
                ),
            ],
            points: vec![
                ([0, 0], -0.02999878),
                ([1, 1], 0.06752014),
                ([7, 7], 0.056243896),
                ([3, 8], 0.06436157),
                ([5, 9], -0.58483887),
                ([7, 15], 0.16088104),
                ([2, 12], -0.15002441),
                ([6, 4], 0.3149414),
            ],
        },
    ];

    for case in block_cases.into_iter().chain(planar_cases) {
        let weight = PackedWeight::try_new(case.format, case.shape).unwrap_or_else(|error| {
            panic!(
                "{}: PackedWeight::try_new({:?}) failed: {error}",
                case.label, case.shape
            )
        });
        let owner = Arc::new(
            PackedPayload::try_new(
                weight,
                case.sources
                    .iter()
                    .map(|(role, bytes)| (*role, Arc::<[u8]>::from(bytes.as_slice()))),
            )
            .unwrap_or_else(|error| {
                panic!("{}: PackedPayload::try_new failed: {error}", case.label)
            }),
        );
        let [out, k] = weight.shape();
        let m = case.points.len();

        let mut activation_data = vec![0.0f32; m * k];
        for (row, &([_, kcol], _)) in case.points.iter().enumerate() {
            activation_data[row * k + kcol] = 1.0;
        }
        let output = contract_through_the_oracle(
            case.label,
            &owner,
            HostTensor::f32(vec![m, k], activation_data),
        )?;
        for (row, &([orow, kcol], expected)) in case.points.iter().enumerate() {
            let got = output.as_f32().unwrap()[row * out + orow];
            // "A dense matmul over the literal expected weight" (SC-001), computed through the same
            // `crate::matmul` the graph's own MatMul op runs, over a one-hot `[1,1]` activation and the
            // literal itself as the sole `[1,1]` weight - never a second call of the decoder. This is
            // deliberately not a plain `expected.to_bits()` comparison: the accumulator both sides share
            // starts at `+0.0` and IEEE 754 fixes `+0.0 + -0.0 = +0.0`, so a `-0.0` literal (Q3_K index
            // 121, for one) legitimately comes out of ANY sum-based matmul, packed or dense alike, as
            // `+0.0` - comparing to the raw literal bit pattern would fail on a correct oracle.
            // Inlined `poot_eval::ops::contraction::matmul`'s 1x1x1 case directly (that function is
            // `pub(crate)` to poot-eval, unreachable from this external integration test): a K=1
            // dense matmul's accumulator starts at `+0.0` and adds exactly one `1.0 * expected` term.
            let expected_matmul: f32 = 0.0f32 + 1.0 * expected;
            assert_eq!(
                got.to_bits(),
                expected_matmul.to_bits(),
                "{} [{orow},{kcol}]: got {got:?}, expected (dense matmul over literal {expected:?}) {expected_matmul:?}",
                case.label
            );
        }
    }
    Ok(())
}

/// A GGUF block-format payload of `shape` with deterministic pseudo-random blocks whose f16
/// factors are all finite.
fn gguf_payload(format: WeightFormat, shape: [usize; 2]) -> Arc<PackedPayload> {
    let Storage::Blocks(layout) = format.descriptor().storage else {
        unreachable!("GGUF formats are block formats")
    };
    let descriptor = PackedWeight::try_new(format, shape).unwrap();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut bytes: Vec<u8> = (0..descriptor.source_bytes(SourceRole::Blocks))
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect();
    // Keep every f16 factor (Q8_0 `d`; Q4_K `d`, `dmin`) finite: clear its top exponent bit.
    for block in bytes.chunks_mut(layout.bytes) {
        for role in [OperandRole::Scale, OperandRole::Min] {
            if let Some(field) = layout.field(role) {
                let byte = (field.field.pieces[0].layout.bit(0) / 8) as usize;
                block[byte + 1] &= !0x40;
            }
        }
    }
    Arc::new(PackedPayload::try_new(descriptor, [(SourceRole::Blocks, Arc::from(bytes))]).unwrap())
}

/// SC-003 (dquant.md R5): the CPU oracle decodes each weight row once, so its decode work is
/// linear in `K`, through both oracle entries (`PackedDequant` and `PackedContraction`). Observed
/// without a production hook, through the test binary's allocation counter: a row decode writes into
/// the oracle's own row buffer, while a per-element decode materializes a whole block-format row for
/// every value, so the oracle's allocation grows from `O(out * K)` to `O(out * K * K)` bytes and
/// breaks the linear bound at both widths. Mutation (card 642, before the per-element
/// `PackedPayload::decode` was deleted as test-only by card 652): `decode_packed_row` reverted to a
/// per-element `owner.decode([row, c])` loop; Q8_0 `PackedDequant` at K=512 allocated 4211363 bytes
/// against the 32768 budget (green: 17059).
#[test]
fn packed_oracle_decode_work_is_linear_in_k() -> Result<(), BuilderAppendError> {
    // Generous per-logical-value budget for a linear oracle (output, row buffer, bookkeeping); the
    // per-element path spends `4 * K` bytes per value, over 100x this at both widths.
    const LINEAR_BYTES_PER_VALUE: usize = 16;
    let out = 4;
    for format in [WeightFormat::Q8_0, WeightFormat::Q4_K] {
        for k in [512, 1024] {
            let packed_inputs = |graph: &Graph, owner: &Arc<PackedPayload>| {
                graph
                    .inputs
                    .iter()
                    .filter_map(|&input| {
                        let source = graph
                            .meta(input)
                            .name
                            .as_deref()
                            .and_then(PackedSourceName::parse)?;
                        Some((
                            input,
                            Value::from(PackedComponentRef::new(Arc::clone(owner), source.role())),
                        ))
                    })
                    .collect::<HashMap<_, _>>()
            };

            let owner = gguf_payload(format, [out, k]);
            let descriptor = owner.weight();
            let b = Builder::new();
            let [(name, source_type)] = <[(PackedSourceName, TensorType); 1]>::try_from(
                packed_source_constants("layer", descriptor),
            )
            .expect("a GGUF block format has exactly one source");
            let source = b.constant(name.as_str(), source_type);
            let decoded = b.packed_dequant(&[source], descriptor);
            let graph = b.finish(decoded);
            let inputs = packed_inputs(&graph, &owner);
            let (result, dequant_bytes) = allocated_bytes(|| {
                eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).map(|r| r.output)
            });
            assert!(matches!(result.unwrap(), Value::Host(_)));

            let b = Builder::new();
            let activation = b.slot_named(Slot::Activation, "work", TensorType::f32(vec![2, k]));
            let output = packed_linear(&b, activation, "layer", descriptor, None, None)?;
            let graph = prepare_packed_dequant_production(&b.finish(output)).unwrap();
            let mut inputs = packed_inputs(&graph, &owner);
            inputs.insert(
                activation.id,
                Value::Host(HostTensor::f32(vec![2, k], vec![0.5; 2 * k])),
            );
            let (result, contraction_bytes) = allocated_bytes(|| {
                eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).map(|r| r.output)
            });
            assert!(matches!(result.unwrap(), Value::Host(_)));

            let budget = LINEAR_BYTES_PER_VALUE * out * k;
            for (entry, bytes) in [
                ("PackedDequant", dequant_bytes),
                ("PackedContraction", contraction_bytes),
            ] {
                assert!(
                    bytes <= budget,
                    "{format:?} {entry} K={k}: the oracle allocated {bytes} bytes, over the linear \
                     budget {budget} (a per-element decode is quadratic in K)"
                );
            }
        }
    }
    Ok(())
}

/// Card 642 (dquant.md R5): the CPU oracle's `PackedContraction` and `PackedDequant`
/// on GGUF block formats (one `Blocks` carrier) decode through `PackedPayload::decode_row`, and still
/// equal the scalar definition bit for bit: every output is `sum over k of x[r,k] * decode([o,k])`,
/// summed in K order from `0.0` with the per-element decoder. Q8_0 and Q4_K (multi-super-block rows),
/// dense and block-diagonal (`blocks = 2`). Mutation: decode weight row `output` instead of
/// `block * block_out + output` in `evaluate_packed_contraction`; the block-diagonal Q8_0 case goes
/// red.
#[test]
fn packed_contraction_oracle_decodes_gguf_rows_bit_exactly() -> Result<(), BuilderAppendError> {
    for (format, blocks) in [
        (WeightFormat::Q8_0, 1),
        (WeightFormat::Q8_0, 2),
        (WeightFormat::Q4_K, 1),
    ] {
        let (out, k) = (4, 512);
        let owner = gguf_payload(format, [out, k]);
        let descriptor = owner.weight();
        let rows_per_block = 3;
        let activation_shape = if blocks == 1 {
            vec![rows_per_block, k]
        } else {
            vec![blocks, rows_per_block, k]
        };
        let b = Builder::new();
        let activation = b.slot_named(
            Slot::Activation,
            "gguf-row-test",
            TensorType::f32(activation_shape.clone()),
        );
        let output = if blocks == 1 {
            packed_linear(&b, activation, "gguf", descriptor, None, None)?
        } else {
            packed_block_diagonal_linear(&b, activation, "gguf", descriptor, blocks, None)?
        };
        let graph = prepare_packed_dequant_production(&b.finish(output)).unwrap();
        assert!(matches!(
            graph.eqns.as_slice(),
            [poot_graph_ir::Eqn {
                op: OpKind::PackedContraction { .. },
                ..
            }]
        ));
        let x: Vec<f32> = (0..activation_shape.iter().product())
            .map(|index: usize| ((index * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        let mut inputs = HashMap::new();
        inputs.insert(
            activation.id,
            Value::Host(HostTensor::f32(activation_shape.clone(), x.clone())),
        );
        for graph_input in &graph.inputs {
            if let Some(source) = graph
                .meta(*graph_input)
                .name
                .as_deref()
                .and_then(PackedSourceName::parse)
            {
                inputs.insert(
                    *graph_input,
                    PackedComponentRef::new(Arc::clone(&owner), source.role()).into(),
                );
            }
        }
        let Value::Host(actual) = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| r.output)
            .unwrap()
        else {
            panic!("dense result");
        };
        let block_out = out / blocks;
        let mut weight_row = vec![0.0f32; k];
        for block in 0..blocks {
            for r in 0..rows_per_block {
                let row = block * rows_per_block + r;
                for o in 0..block_out {
                    owner
                        .decode_row(block * block_out + o, &mut weight_row)
                        .unwrap();
                    let mut expected = 0.0f32;
                    for kk in 0..k {
                        expected += x[row * k + kk] * weight_row[kk];
                    }
                    let got = actual.as_f32().unwrap()[row * block_out + o];
                    assert_eq!(
                        got.to_bits(),
                        expected.to_bits(),
                        "{format:?} blocks={blocks} [{row},{o}]: {got} vs {expected}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Card 545a: the CPU oracle evaluates a claimed `PackedRowGather` by decoding only
/// the gathered rows (uncapped: a table far past `PACKED_ORACLE_MAX_ELEMENTS` still evaluates), each
/// equal bit for bit to the payload's row decoder; an id past the table is a typed refusal. Mutation:
/// decode row `id + 1` in `evaluate_packed_row_gather`; the row comparison goes red.
#[test]
fn packed_row_gather_oracle_decodes_only_the_gathered_rows() {
    use super::recognize_packed_row_gathers;
    // 8M+ logical values: a full materialization would exceed the oracle cap.
    let (rows, k) = (33_000, 256);
    assert!(rows * k > PACKED_ORACLE_MAX_ELEMENTS);
    let descriptor = PackedWeight::try_new(WeightFormat::Q8_0, [rows, k]).unwrap();
    let block = {
        let mut block = vec![0u8; 34];
        block[..2].copy_from_slice(&[0x00, 0x3c]);
        for (i, byte) in block[2..].iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(7);
        }
        block
    };
    let mut bytes = Vec::with_capacity(descriptor.source_bytes(SourceRole::Blocks));
    for row in 0..rows {
        for b in 0..k / 32 {
            let mut blk = block.clone();
            blk[2] = (row % 251) as u8;
            blk[3] = b as u8;
            bytes.extend_from_slice(&blk);
        }
    }
    let owner = Arc::new(
        PackedPayload::try_new(descriptor, [(SourceRole::Blocks, Arc::from(bytes))]).unwrap(),
    );
    let b = Builder::new();
    let ids = b.slot(Slot::Token, TensorType::f32(vec![3]));
    let out = poot_graph_ir::ops::packed_embedding(&b, ids, "embed", descriptor).unwrap();
    let graph = recognize_packed_row_gathers(&b.finish(out));
    assert!(matches!(
        graph.eqns.as_slice(),
        [poot_graph_ir::Eqn {
            op: OpKind::PackedRowGather { .. },
            ..
        }]
    ));
    let mut inputs: HashMap<usize, Value> = HashMap::new();
    for &id in &graph.inputs {
        if let Some(source) = graph
            .meta(id)
            .name
            .as_deref()
            .and_then(PackedSourceName::parse)
        {
            inputs.insert(
                id,
                PackedComponentRef::new(Arc::clone(&owner), source.role()).into(),
            );
        }
    }
    let wanted = [32_999usize, 0, 250];
    inputs.insert(
        ids.id,
        HostTensor::f32(vec![3], wanted.iter().map(|&r| r as f32).collect()).into(),
    );
    let Value::Host(got) = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| r.output)
        .unwrap()
    else {
        panic!("dense result");
    };
    assert_eq!(got.shape(), vec![3, k]);
    let mut expected = vec![0.0f32; k];
    for (r, &row) in wanted.iter().enumerate() {
        owner.decode_row(row, &mut expected).unwrap();
        for (c, (g, e)) in got.as_f32().unwrap()[r * k..(r + 1) * k]
            .iter()
            .zip(&expected)
            .enumerate()
        {
            assert_eq!(g.to_bits(), e.to_bits(), "row {row} col {c}: {g} vs {e}");
        }
    }
    inputs.insert(
        ids.id,
        HostTensor::f32(vec![3], vec![0.0, rows as f32, 1.0]).into(),
    );
    assert!(matches!(
        eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).map(|r| r.output),
        Err(EvalError::Packed(PackedEvalError::Binding {
            field: "row_ids.range"
        }))
    )); // Review minor 12: a NaN or fractional id names no row and is refused, never rounded.
    for bad in [f32::NAN, 1.5] {
        inputs.insert(ids.id, HostTensor::f32(vec![3], vec![0.0, bad, 1.0]).into());
        assert!(
            matches!(
                eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).map(|r| r.output),
                Err(EvalError::Packed(PackedEvalError::Binding {
                    field: "row_ids.range"
                }))
            ),
            "id {bad}"
        );
    }
}

/// A Q8_0 `[rows, 256]` payload whose every row is distinct (the row index lands in each block).
fn distinct_row_q8_0(rows: usize, k: usize) -> Arc<PackedPayload> {
    let descriptor = PackedWeight::try_new(WeightFormat::Q8_0, [rows, k]).unwrap();
    let mut bytes = Vec::with_capacity(descriptor.source_bytes(SourceRole::Blocks));
    for row in 0..rows {
        for b in 0..k / 32 {
            let mut block = vec![0u8; 34];
            block[..2].copy_from_slice(&[0x00, 0x3c]);
            for (i, byte) in block[2..].iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(7);
            }
            block[2] = (row % 251) as u8;
            block[3] = b as u8;
            bytes.extend_from_slice(&block);
        }
    }
    Arc::new(PackedPayload::try_new(descriptor, [(SourceRole::Blocks, Arc::from(bytes))]).unwrap())
}

fn bind_carriers(graph: &Graph, owner: &Arc<PackedPayload>) -> HashMap<usize, Value> {
    graph
        .inputs
        .iter()
        .filter_map(|&id| {
            let source = graph
                .meta(id)
                .name
                .as_deref()
                .and_then(PackedSourceName::parse)?;
            Some((
                id,
                PackedComponentRef::new(Arc::clone(owner), source.role()).into(),
            ))
        })
        .collect()
}

/// Card 545a SC-006, over one weight whose `[out, K]` exceeds the historical
/// `PACKED_ORACLE_MAX_ELEMENTS`: a packed projection (claimed `PackedContraction`) evaluates through
/// the row decode, uncapped, each output equal bit for bit to the dot product of the activation with
/// the decoded row.
///
/// Mutation (recorded, never left in the tree): re-adding the cap to the row decode
/// (`check_packed_oracle_work(rows * block_out, k)` in `evaluate_packed_contraction`) turns the
/// projection row red with `OracleLimit { resource: "elements", requested: 8448000, limit: 8388608 }`.
///
/// Card 554d FLAG: this test used to have a second half asserting that materializing the same weight
/// (a bare `PackedDequant` as the output) is refused with the fixed cap's `OracleLimit` error. That
/// fixed cap is gone (`ops/packed.rs`: the packed oracle's allocation limit is now an opt-in
/// `EvalBudget` the caller states per call); with this file's calls stating `EvalBudget::
/// UNBOUNDED`, the same materialization now succeeds instead of erroring. Removed rather than
/// improvised - which `EvalBudget` the materialization path should be bound to, if any, is a
/// deliberate call-site decision; see this crate's migration report.
#[test]
fn a_packed_projection_past_the_oracle_cap_evaluates() {
    use super::recognize_packed_contractions;
    let (out, k) = (33_000, 256);
    assert!(out * k > PACKED_ORACLE_MAX_ELEMENTS);
    let owner = distinct_row_q8_0(out, k);

    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![1, k]));
    let y = packed_linear(&b, x, "proj", owner.weight(), None, None).unwrap();
    let graph = recognize_packed_contractions(&b.finish(y));
    assert!(
        graph
            .eqns
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::PackedContraction { .. }))
    );
    let activation: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();
    let mut inputs = bind_carriers(&graph, &owner);
    inputs.insert(x.id, HostTensor::f32(vec![1, k], activation.clone()).into());
    let Value::Host(got) = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| r.output)
        .expect("the projection is uncapped")
    else {
        panic!("dense result");
    };
    assert_eq!(got.as_f32().unwrap().len(), out);
    let mut row = vec![0.0f32; k];
    for o in [0usize, 250, 32_999] {
        owner.decode_row(o, &mut row).unwrap();
        let mut expected = 0.0f32;
        for (a, w) in activation.iter().zip(&row) {
            expected += a * w;
        }
        assert_eq!(
            got.as_f32().unwrap()[o].to_bits(),
            expected.to_bits(),
            "row {o}"
        );
    }
    // Card 554d FLAG: the "full materialization past the cap is refused" half of this test was here
    // (bind `packed_source_constants`/`packed_dequant` over the same `owner.weight()`, then expect
    // `Err(EvalError::Packed(PackedEvalError::OracleLimit { resource: "elements", .. }))`). Removed;
    // see the doc comment above this function.
}

/// SC-008 (first clause): a `PackedContraction` above `EvalBudget::bounded(work, bytes)` is
/// `EvalError::Budget { needed, limit }` with the literal element count; the same equation under
/// `EvalBudget::UNBOUNDED` evaluates.
///
/// Mutation: drop the `opts.charge(...)` call on `OpKind::PackedContraction` in
/// `walk.rs::evaluate_equation`; the bounded row stops erroring and this test goes red.
#[test]
fn packed_contraction_above_budget_is_refused_unbounded_evaluates() -> Result<(), BuilderAppendError>
{
    let owner = payload(WeightFormat::E2m1Row32, [3, 35]);
    let descriptor = owner.weight();
    let builder = Builder::new();
    let activation = builder.slot_named(
        Slot::Activation,
        "budget-candidate",
        TensorType::f32(vec![2, 35]),
    );
    let output = packed_linear(&builder, activation, "budget.layer", descriptor, None, None)?;
    let source = builder.finish(output);
    let direct = prepare_packed_dequant_production(&source).unwrap();
    assert!(matches!(
        direct.eqns.as_slice(),
        [poot_graph_ir::Eqn {
            op: poot_graph_ir::OpKind::PackedContraction { .. },
            ..
        }]
    ));

    let candidate = &direct.eqns[0];
    let [
        _,
        poot_graph_ir::Operand::Value(weight_id),
        poot_graph_ir::Operand::Value(scale_id),
    ] = candidate.inputs.as_slice()
    else {
        panic!("candidate must retain activation, weight, and scale values");
    };
    let weight =
        PackedComponentRef::new(Arc::clone(&owner), SourceRole::Planar(OperandRole::Codes));
    let scale = PackedComponentRef::new(Arc::clone(&owner), SourceRole::Planar(OperandRole::Scale));
    let inputs = HashMap::from([
        (
            activation.id,
            Value::Host(HostTensor::f32(
                vec![2, 35],
                (0..70).map(|index| index as f32 / 17.0).collect(),
            )),
        ),
        (*weight_id, Value::Packed(weight)),
        (*scale_id, Value::Packed(scale)),
    ]);

    // The output is [2, 3]: 6 elements, 24 bytes. A 3-element ceiling must refuse before evaluating.
    let bounded = eval(
        &direct,
        &inputs,
        EvalOptions::new(EvalBudget::bounded(3, 1_000_000)),
    )
    .expect_err("a PackedContraction past max_work_elements must refuse");
    assert!(
        matches!(
            bounded,
            EvalError::Budget {
                resource: "elements",
                needed: 6,
                limit: 3,
                ..
            }
        ),
        "{bounded:?}"
    );

    let unbounded = eval(&direct, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("UNBOUNDED must still evaluate the same contraction");
    assert!(matches!(unbounded.output, Value::Host(_)));
    Ok(())
}
