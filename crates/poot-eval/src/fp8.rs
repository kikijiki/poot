//! Exact OCP E4M3FN CPU semantics and the Stage 1 portable packed-u32 storage oracle.
//!
//! The raw row-major bytes of an E4M3FN [`HostTensor`] are authoritative.

use poot_tensor::{DType, HostData, HostTensor};

use poot_load::packed_safetensors::ExactSourceKind;
use poot_quant::scalar::e4m3fn_to_f32;

use crate::EvalError;
use crate::ops::index_rule::{IndexValue, index_at};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Fp8Error {
    #[error("e4m3fn payload for shape {shape:?} needs {expected} elements, got {got}")]
    ElementCount {
        shape: Vec<usize>,
        expected: usize,
        got: usize,
    },
    #[error(transparent)]
    Carrier(#[from] poot_tensor::CarrierError),
    #[error(
        "e4m3fn tensor needs an authenticated F8_E4M3 source owner, got {kind:?} with dtype {dtype}"
    )]
    SourceOwnerKind {
        kind: ExactSourceKind,
        dtype: String,
    },
}

/// Static layout of the normative Stage 1 portable representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortableE4m3FnLayout {
    pub outer: usize,
    pub logical_row_len: usize,
    pub words_per_row: usize,
    pub word_count: usize,
    pub device_bytes: usize,
    pub host_bytes: usize,
}

impl PortableE4m3FnLayout {
    pub fn for_shape(shape: &[usize]) -> Self {
        let logical_row_len = shape.last().copied().unwrap_or(1);
        let outer = if shape.is_empty() {
            1
        } else {
            shape[..shape.len() - 1].iter().product()
        };
        let words_per_row = logical_row_len.div_ceil(4);
        let word_count = outer.saturating_mul(words_per_row);
        let host_bytes = shape.iter().product();
        PortableE4m3FnLayout {
            outer,
            logical_row_len,
            words_per_row,
            word_count,
            device_bytes: word_count.saturating_mul(4),
            host_bytes,
        }
    }
}

/// An E4M3FN [`HostTensor`] over `bytes` (authoritative row-major bytes, unpadded): the one
/// constructor of that carrier, refusing a payload that is not `shape`'s element count.
pub fn e4m3fn_tensor(shape: Vec<usize>, bytes: Vec<u8>) -> Result<HostTensor, Fp8Error> {
    validate_elements(&shape, bytes.len())?;
    Ok(HostTensor::new(
        DType::E4M3FN,
        shape,
        HostData::Bytes(bytes.into()),
    )?)
}

/// Encode `values` as an E4M3FN [`HostTensor`] (round-to-nearest-even, saturating to finite).
pub fn encode_e4m3fn_tensor(shape: Vec<usize>, values: &[f32]) -> Result<HostTensor, Fp8Error> {
    validate_elements(&shape, values.len())?;
    let bytes = values.iter().copied().map(encode_e4m3fn).collect();
    e4m3fn_tensor(shape, bytes)
}

/// Pack each final-axis row of an E4M3FN tensor independently into little-endian u32 lanes. Newly
/// allocated words start at zero, so all missing lanes in a ragged row remain deterministic zero
/// padding. Test-only (card 546b): the production packer was `GpuExecutor::bind_resident`'s E4M3
/// upload path, gone with the rest of the pre-contract executor.
#[cfg(test)]
pub(crate) fn pack_portable_words(tensor: &HostTensor) -> Vec<u32> {
    let layout = PortableE4m3FnLayout::for_shape(tensor.shape());
    let bytes = tensor.view().bytes();
    let mut words = vec![0u32; layout.word_count];
    for row in 0..layout.outer {
        for col in 0..layout.logical_row_len {
            let byte = bytes[row * layout.logical_row_len + col] as u32;
            let word = row * layout.words_per_row + col / 4;
            words[word] |= byte << (8 * (col % 4));
        }
    }
    words
}

fn validate_elements(shape: &[usize], got: usize) -> Result<(), Fp8Error> {
    let expected = shape.iter().product();
    if got == expected {
        Ok(())
    } else {
        Err(Fp8Error::ElementCount {
            shape: shape.to_vec(),
            expected,
            got,
        })
    }
}

/// Encode one f32 as OCP E4M3FN with round-to-nearest-even and saturate-to-finite behavior.
pub fn encode_e4m3fn(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude.is_infinite() || magnitude >= 448.0 {
        return sign | 0x7e;
    }

    // Search the 127 nonnegative finite values; f64 distance is exact for these dyadic f32 values.
    // On an exact midpoint the byte's low mantissa bit selects the even significand.
    let mut best = 0u8;
    let mut best_distance = magnitude as f64;
    for code in 1u8..=0x7e {
        let candidate = e4m3fn_to_f32(code) as f64;
        let distance = (candidate - magnitude as f64).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0 && best & 1 != 0)
        {
            best = code;
            best_distance = distance;
        }
    }
    sign | best
}

/// Validate a runtime `DynamicUpdateSlice` start through the one index rule
/// ([`crate::ops::index_rule::index_at`]): `index` must select a start in
/// `0..=(operand_shape[axis] - update_shape[axis])`, the range of starts whose window fits (deval 4).
/// `operand_shape`/`update_shape` come from the graph's own avals, already shape-compatible by
/// `infer`'s `ShapeError::UpdateSlice` check; an incompatible pair (never reachable from a real graph)
/// is still refused instead of underflowing. The one device-side pre-flight (PTX capture/replay) that
/// needs this outside the oracle walk itself.
pub fn validate_dynamic_update_runtime_index(
    operand_shape: &[usize],
    update_shape: &[usize],
    axis: usize,
    index: f32,
    eqn: poot_graph_ir::ValueId,
) -> Result<usize, EvalError> {
    let starts = operand_shape
        .get(axis)
        .zip(update_shape.get(axis))
        .and_then(|(&operand_extent, &update_extent)| operand_extent.checked_sub(update_extent))
        .map(|slack| slack + 1)
        .ok_or_else(|| {
            EvalError::unsupported(
                "dynamic_update_slice",
                format!(
                    "axis {axis} update extent exceeds operand extent (operand {operand_shape:?}, update {update_shape:?})"
                ),
            )
        })?;
    index_at(IndexValue::F32(index), 0, starts, eqn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use poot_graph_ir::{Builder, GraphValidationError, StateRole, TensorType};

    fn reference_decode(bits: u8) -> f32 {
        let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = (bits >> 3) & 0x0f;
        let mantissa = bits & 0x07;
        if exponent == 0x0f && mantissa == 0x07 {
            return f32::NAN;
        }
        if exponent == 0 {
            return sign * (mantissa as f32 / 8.0) * 2.0f32.powi(-6);
        }
        sign * (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
    }

    #[test]
    fn exhaustive_decode_matches_independent_bit_reference() {
        for bits in 0u8..=u8::MAX {
            let got = e4m3fn_to_f32(bits);
            let expected = reference_decode(bits);
            if expected.is_nan() {
                assert!(got.is_nan(), "bits {bits:#04x}");
                assert_eq!(
                    got.to_bits(),
                    0x7fc0_0000,
                    "the canonical positive quiet NaN"
                );
            } else {
                assert_eq!(got.to_bits(), expected.to_bits(), "bits {bits:#04x}");
            }
        }
    }

    #[test]
    fn encode_specials_boundaries_and_all_halfway_ties() {
        assert_eq!(encode_e4m3fn(0.0), 0x00);
        assert_eq!(encode_e4m3fn(-0.0), 0x80);
        assert_eq!(encode_e4m3fn(f32::INFINITY), 0x7e);
        assert_eq!(encode_e4m3fn(f32::NEG_INFINITY), 0xfe);
        assert_eq!(encode_e4m3fn(f32::NAN), 0x7f);
        assert_eq!(encode_e4m3fn(f32::from_bits(0xffc1_2345)), 0x7f);
        assert_eq!(encode_e4m3fn(1.0 / 512.0), 0x01);
        assert_eq!(encode_e4m3fn(448.0), 0x7e);
        assert_eq!(encode_e4m3fn(449.0), 0x7e);
        assert_eq!(encode_e4m3fn(-449.0), 0xfe);

        for lower in 0u8..0x7e {
            let upper = lower + 1;
            let midpoint = (e4m3fn_to_f32(lower) + e4m3fn_to_f32(upper)) * 0.5;
            let expected = if lower & 1 == 0 { lower } else { upper };
            let below = f32::from_bits(midpoint.to_bits() - 1);
            let above = f32::from_bits(midpoint.to_bits() + 1);
            assert_eq!(encode_e4m3fn(below), lower, "below midpoint {midpoint}");
            assert_eq!(
                encode_e4m3fn(midpoint),
                expected,
                "tie between {lower:#04x} and {upper:#04x}"
            );
            assert_eq!(encode_e4m3fn(above), upper, "above midpoint {midpoint}");
        }
    }

    #[test]
    fn every_finite_byte_round_trips_and_nan_canonicalizes() {
        for bits in 0u8..=u8::MAX {
            let decoded = e4m3fn_to_f32(bits);
            if decoded.is_nan() {
                assert_eq!(encode_e4m3fn(decoded), 0x7f);
            } else {
                assert_eq!(encode_e4m3fn(decoded), bits, "bits {bits:#04x}");
            }
        }
    }

    #[test]
    fn raw_payload_and_portable_row_packing_are_exact() {
        let scalar = e4m3fn_tensor(vec![], vec![0x81]).unwrap();
        assert_eq!(
            PortableE4m3FnLayout::for_shape(scalar.shape()).device_bytes,
            4
        );
        assert_eq!(pack_portable_words(&scalar), vec![0x0000_0081]);

        let aligned = e4m3fn_tensor(vec![2, 4], (0u8..8).collect()).unwrap();
        assert_eq!(
            PortableE4m3FnLayout::for_shape(aligned.shape()).device_bytes,
            8
        );
        assert_eq!(
            pack_portable_words(&aligned),
            vec![0x0302_0100, 0x0706_0504]
        );

        let bytes: Vec<u8> = (0u8..10).collect();
        let raw = e4m3fn_tensor(vec![2, 5], bytes.clone()).unwrap();
        let words = pack_portable_words(&raw);
        assert_eq!(PortableE4m3FnLayout::for_shape(raw.shape()).host_bytes, 10);
        assert_eq!(
            PortableE4m3FnLayout::for_shape(raw.shape()).device_bytes,
            16
        );
        assert_eq!(
            words,
            vec![0x0302_0100, 0x0000_0004, 0x0807_0605, 0x0000_0009]
        );
    }

    #[test]
    fn host_reshape_preserves_authoritative_bytes_and_changes_packed_rows() {
        let bytes: Vec<u8> = (0u8..10).collect();
        let source = e4m3fn_tensor(vec![2, 5], bytes.clone()).unwrap();
        let reshaped = source.reshaped(vec![5, 2]).unwrap();
        assert_eq!(reshaped.view().bytes(), bytes);
        assert_eq!(
            PortableE4m3FnLayout::for_shape(source.shape()).device_bytes,
            16
        );
        assert_eq!(
            PortableE4m3FnLayout::for_shape(reshaped.shape()).device_bytes,
            20
        );
        assert_ne!(pack_portable_words(&source), pack_portable_words(&reshaped));
    }

    #[test]
    fn generic_graph_eval_rejects_fp8_bound_as_a_plain_f32_tensor() {
        // Card 554d: there is no more a separate "generic" walk distinct from an
        // fp8-aware one - the one walk handles every E4M3FN-touching graph itself, so
        // `Cast(F32, E4M3FN)` with a legitimately F32-bound input is no longer refused outright; it
        // properly encodes (`generic_value_cast_and_reshape_preserve_fp8_storage` already covers
        // this). What the one walk still refuses, categorically, is binding an E4M3FN-*declared*
        // input as a plain `Value::Host` - reinterpreting its f32 bit pattern as fp8 bytes (or vice
        // versa) instead of carrying authoritative raw bytes through `Value::E4M3Fn`.
        let builder = Builder::new();
        let x = builder.constant("x", TensorType::new(vec![2], DType::E4M3FN));
        let graph = builder.finish(x);
        assert!(
            graph.eqns.is_empty(),
            "regression requires an identity graph"
        );
        let inputs: HashMap<_, crate::Value> = HashMap::from([(
            graph.inputs[0],
            poot_tensor::HostTensor::f32(vec![2], vec![1.0, 2.0]).into(),
        )]);
        let message = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .unwrap_err()
        .to_string();
        assert!(message.contains("e4m3fn"), "{message}");
        assert!(message.contains("Value::Host"), "{message}");
    }

    #[test]
    fn generic_graph_eval_casts_a_legitimately_bound_f32_tensor_to_e4m3fn() {
        // The positive case `generic_graph_eval_rejects_fp8_bound_as_a_plain_f32_tensor` deliberately
        // does not cover: an F32-*declared* input, bound as an ordinary `Value::Host`, cast to
        // E4M3FN. This is a real encode, not a reinterpretation, and the one walk computes it.
        let builder = Builder::new();
        let x = builder.constant("x", TensorType::f32(vec![2]));
        let y = builder.cast(x, DType::E4M3FN);
        let graph = builder.finish(y);
        let inputs: HashMap<_, crate::Value> = HashMap::from([(
            graph.inputs[0],
            poot_tensor::HostTensor::f32(vec![2], vec![1.0, 2.0]).into(),
        )]);
        let output = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .unwrap()
        .output;
        let crate::Value::Host(tensor) = output else {
            panic!("expected an E4M3Fn result, got {output:?}")
        };
        assert_eq!(
            tensor.view().bytes(),
            [encode_e4m3fn(1.0), encode_e4m3fn(2.0)],
            "the cast must encode, not reinterpret, the f32 bits"
        );
    }

    #[test]
    fn generic_value_cast_and_reshape_preserve_fp8_storage() {
        let builder = Builder::new();
        let x = builder.constant("x", TensorType::new(vec![2, 5], DType::E4M3FN));
        let reshaped = builder.reshape(x, vec![5, 2]);
        let graph = builder.finish(reshaped);
        let bytes = vec![0x00, 0x80, 0x01, 0x7e, 0x7f, 0xff, 0x38, 0xb8, 0x55, 0xd5];
        let inputs = HashMap::from([(
            x.id,
            crate::Value::Host(e4m3fn_tensor(vec![2, 5], bytes.clone()).unwrap()),
        )]);
        let got = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap();
        let crate::Value::Host(got) = got else {
            panic!("reshape must retain e4m3fn storage")
        };
        assert_eq!(got.shape(), [5, 2]);
        assert_eq!(got.view().bytes(), bytes);

        let builder = Builder::new();
        let x = builder.constant("x", TensorType::f32(vec![2, 5]));
        let encoded = builder.cast(x, DType::E4M3FN);
        let decoded = builder.cast(encoded, DType::F32);
        let graph = builder.finish(decoded);
        let values = vec![
            0.0,
            -0.0,
            1.0 / 512.0,
            1.0,
            -1.0,
            448.0,
            -449.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.3,
        ];
        let inputs = HashMap::from([(
            x.id,
            crate::Value::Host(poot_tensor::HostTensor::f32(vec![2, 5], values.clone())),
        )]);
        let got = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap();
        let crate::Value::Host(got) = got else {
            panic!("decode must return dense f32 storage")
        };
        let expected: Vec<f32> = values
            .into_iter()
            .map(encode_e4m3fn)
            .map(e4m3fn_to_f32)
            .collect();
        assert_eq!(got.as_f32().unwrap(), expected);
    }

    #[test]
    fn generic_value_same_dtype_e4m3fn_cast_aliases_authoritative_bytes() {
        let builder = Builder::new();
        let x = builder.constant("raw", TensorType::new(vec![2, 5], DType::E4M3FN));
        let same_dtype = builder.cast(x, DType::E4M3FN);
        let graph = builder.finish(same_dtype);
        let bytes = vec![0x00, 0x80, 0x01, 0x7e, 0x7f, 0xff, 0x38, 0xb8, 0x55, 0xd5];
        let raw = e4m3fn_tensor(vec![2, 5], bytes.clone()).unwrap();
        let raw_bytes = raw.view().bytes().as_ptr();
        let inputs = HashMap::from([(x.id, crate::Value::Host(raw))]);

        let got = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap();
        let crate::Value::Host(got) = got else {
            panic!("same-dtype Cast must retain e4m3fn storage")
        };
        assert_eq!(got.view().bytes(), bytes);
        assert_eq!(
            got.view().bytes().as_ptr(),
            raw_bytes,
            "Cast copied raw bytes"
        );
    }

    #[test]
    fn generic_value_zero_eqn_raw_identity_returns_input_without_reencoding() {
        let builder = Builder::new();
        let x = builder.constant("raw", TensorType::new(vec![2, 5], DType::E4M3FN));
        let graph = builder.finish(x);
        assert!(
            graph.eqns.is_empty(),
            "identity path must have zero equations"
        );
        let bytes = vec![0x00, 0x80, 0x01, 0x7e, 0x7f, 0xff, 0x38, 0xb8, 0x55, 0xd5];
        let raw = e4m3fn_tensor(vec![2, 5], bytes.clone()).unwrap();
        let raw_bytes = raw.view().bytes().as_ptr();
        let inputs = HashMap::from([(x.id, crate::Value::Host(raw))]);

        let got = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap();
        assert_eq!(got.physical_bytes(), 16);
        let crate::Value::Host(got) = got else {
            panic!("zero-equation identity must return e4m3fn storage")
        };
        assert_eq!(got.view().bytes(), bytes);
        assert_eq!(
            got.view().bytes().as_ptr(),
            raw_bytes,
            "identity copied raw bytes"
        );
    }

    #[test]
    fn generic_value_state_pass_through_keeps_authoritative_host_allocation() {
        let builder = Builder::new();
        let state = builder.state_input(
            "raw_state",
            TensorType::new(vec![2, 5], DType::E4M3FN),
            StateRole::Recurrent,
        );
        let graph = builder.finish_with_state(state, &[(state, state)]);
        let bytes = vec![0x7f, 0xff, 1, 2, 3, 4, 5, 6, 7, 8];
        let input = e4m3fn_tensor(vec![2, 5], bytes.clone()).unwrap();
        let input_ptr = input.view().bytes().as_ptr();
        let inputs = HashMap::from([(state.id, crate::Value::Host(input))]);

        let result = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .unwrap();
        let (output, carried) = (result.output, result.state);
        assert_eq!(carried.len(), 1);
        assert_eq!(output.physical_bytes(), 16);
        assert_eq!(carried[0].physical_bytes(), 16);
        let crate::Value::Host(output) = output else {
            panic!("state output must retain e4m3fn storage")
        };
        let crate::Value::Host(carried) = &carried[0] else {
            panic!("carried state must retain e4m3fn storage")
        };
        assert_eq!(output.view().bytes(), bytes);
        assert_eq!(output.view().bytes().as_ptr(), input_ptr);
        assert_eq!(carried.view().bytes().as_ptr(), input_ptr);
    }

    #[test]
    fn generic_value_state_carries_two_ragged_raw_updates_without_dense_reconstruction() {
        let builder = Builder::new();
        let state = builder.state_input(
            "raw_state",
            TensorType::new(vec![2, 5], DType::E4M3FN),
            StateRole::Recurrent,
        );
        let update = builder.constant("raw_update", TensorType::new(vec![2, 2], DType::E4M3FN));
        let index = builder.constant("index", TensorType::f32(vec![]));
        let state_out = builder.dynamic_update_slice_dyn(state, update, index, 1);
        let graph = builder.finish_with_state(state_out, &[(state, state_out)]);

        let initial = e4m3fn_tensor(vec![2, 5], vec![0x7f, 1, 2, 3, 4, 0xff, 5, 6, 7, 8]).unwrap();
        let updates = [
            (1.0, vec![0x80, 0x7e, 0xfe, 0x00]),
            (3.0, vec![0xff, 0x7f, 0x38, 0xb8]),
        ];
        let mut carried = crate::Value::Host(initial);
        for (position, raw_update) in updates {
            let inputs = HashMap::from([
                (state.id, carried),
                (
                    update.id,
                    crate::Value::Host(e4m3fn_tensor(vec![2, 2], raw_update).unwrap()),
                ),
                (
                    index.id,
                    crate::Value::Host(poot_tensor::HostTensor::scalar(position)),
                ),
            ]);
            let result = crate::eval(
                &graph,
                &inputs,
                crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
            )
            .unwrap();
            let (output, state_values) = (result.output, result.state);
            assert_eq!(state_values.len(), 1);
            assert_eq!(output, state_values[0]);
            assert_eq!(state_values[0].physical_bytes(), 16);
            let crate::Value::Host(next) = state_values.into_iter().next().unwrap() else {
                panic!("carried state must retain e4m3fn storage")
            };
            assert_eq!(PortableE4m3FnLayout::for_shape(next.shape()).host_bytes, 10);
            for row in 0..2 {
                assert_eq!(
                    pack_portable_words(&next)[row * 2 + 1] >> 8,
                    0,
                    "ragged state row {row} has nonzero padding"
                );
            }
            carried = crate::Value::Host(next);
        }

        let crate::Value::Host(final_state) = carried else {
            unreachable!()
        };
        assert_eq!(
            final_state.view().bytes(),
            [0x7f, 0x80, 0x7e, 0xff, 0x7f, 0xff, 0xfe, 0x00, 0x38, 0xb8]
        );
    }

    #[test]
    fn generic_value_state_rejects_graph_and_binding_dtype_mismatches() {
        let builder = Builder::new();
        let state = builder.state_input(
            "raw_state",
            TensorType::new(vec![2, 5], DType::E4M3FN),
            StateRole::Recurrent,
        );
        let dense = builder.constant("dense", TensorType::f32(vec![2, 5]));
        let malformed = builder.finish_with_state(state, &[(state, dense)]);
        let inputs = HashMap::from([
            (
                state.id,
                crate::Value::Host(e4m3fn_tensor(vec![2, 5], vec![0; 10]).unwrap()),
            ),
            (
                dense.id,
                crate::Value::Host(poot_tensor::HostTensor::f32(vec![2, 5], vec![0.0; 10])),
            ),
        ]);
        assert!(matches!(
            crate::eval(
                &malformed,
                &inputs,
                crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
            ),
            Err(crate::EvalError::InvalidGraph(
                GraphValidationError::StateTypeMismatch {
                    state_input,
                    state_output,
                    ..
                }
            )) if state_input == state.id && state_output == dense.id
        ));

        let builder = Builder::new();
        let state = builder.state_input(
            "raw_state",
            TensorType::new(vec![2, 5], DType::E4M3FN),
            StateRole::Recurrent,
        );
        let graph = builder.finish_with_state(state, &[(state, state)]);
        let wrong = HashMap::from([(
            state.id,
            crate::Value::Host(poot_tensor::HostTensor::f32(vec![2, 5], vec![0.0; 10])),
        )]);
        // Card 554d: `walk::preflight_bindings` catches this the same way for any input (not a
        // state-specific "generic value storage" check): an E4M3FN-declared value bound as a plain
        // `Value::Host` is a typed `EvalError::Input` mismatch, checked before any equation runs.
        let message = crate::eval(
            &graph,
            &wrong,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .unwrap_err()
        .to_string();
        assert!(message.contains("e4m3fn"), "{message}");
        assert!(message.contains("Value::Host"), "{message}");
    }

    #[test]
    fn generic_value_state_rejects_a_state_input_missing_from_graph_binders() {
        let builder = Builder::new();
        let state = builder.state_input(
            "raw_state",
            TensorType::new(vec![2, 5], DType::E4M3FN),
            StateRole::Recurrent,
        );
        let replacement =
            builder.constant("replacement", TensorType::new(vec![2, 5], DType::E4M3FN));
        let mut graph = builder.finish_with_state(replacement, &[(state, replacement)]);
        graph.inputs.retain(|&id| id != state.id);

        let inputs = HashMap::from([(
            replacement.id,
            crate::Value::Host(e4m3fn_tensor(vec![2, 5], vec![0x7f; 10]).unwrap()),
        )]);
        assert!(matches!(
            crate::eval(
                &graph,
                &inputs,
                crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
            ),
            Err(crate::EvalError::InvalidGraph(
                GraphValidationError::ConstNotInput { value }
            )) if value == state.id
        ));
    }

    #[test]
    fn generic_value_transpose_preserves_raw_bytes() {
        let builder = Builder::new();
        let x = builder.constant("raw", TensorType::new(vec![2, 5], DType::E4M3FN));
        let y = builder.transpose(x, vec![1, 0]);
        let graph = builder.finish(y);
        let bytes = vec![0x00, 0x80, 0x01, 0x7e, 0x7f, 0xff, 0x38, 0xb8, 0x55, 0xd5];
        let input = e4m3fn_tensor(vec![2, 5], bytes).unwrap();
        let inputs = HashMap::from([(x.id, crate::Value::Host(input))]);

        let got = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap();
        let crate::Value::Host(got) = got else {
            panic!("Transpose must retain e4m3fn storage")
        };
        assert_eq!(got.shape(), [5, 2]);
        assert_eq!(
            got.view().bytes(),
            [0x00, 0xff, 0x80, 0x38, 0x01, 0xb8, 0x7e, 0x55, 0x7f, 0xd5]
        );
    }

    #[test]
    fn generic_value_slice_preserves_raw_bytes() {
        let builder = Builder::new();
        let x = builder.constant("raw", TensorType::new(vec![2, 5], DType::E4M3FN));
        let y = builder.slice(x, 1, 1, 4);
        let graph = builder.finish(y);
        let bytes = vec![0x00, 0x80, 0x01, 0x7e, 0x7f, 0xff, 0x38, 0xb8, 0x55, 0xd5];
        let input = e4m3fn_tensor(vec![2, 5], bytes).unwrap();
        let inputs = HashMap::from([(x.id, crate::Value::Host(input))]);

        let got = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap();
        let crate::Value::Host(got) = got else {
            panic!("Slice must retain e4m3fn storage")
        };
        assert_eq!(got.shape(), [2, 3]);
        assert_eq!(got.view().bytes(), [0x80, 0x01, 0x7e, 0x38, 0xb8, 0x55]);
    }

    #[test]
    fn generic_value_concat_preserves_raw_bytes() {
        let builder = Builder::new();
        let a = builder.constant("a", TensorType::new(vec![2, 3], DType::E4M3FN));
        let b = builder.constant("b", TensorType::new(vec![2, 2], DType::E4M3FN));
        let output = builder.concat(1, &[a, b]);
        let graph = builder.finish(output);
        let inputs = HashMap::from([
            (
                a.id,
                crate::Value::Host(e4m3fn_tensor(vec![2, 3], vec![0x7f, 1, 2, 3, 4, 5]).unwrap()),
            ),
            (
                b.id,
                crate::Value::Host(e4m3fn_tensor(vec![2, 2], vec![0xff, 6, 7, 8]).unwrap()),
            ),
        ]);
        let crate::Value::Host(got) = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap() else {
            panic!("Concat must retain e4m3fn storage")
        };
        assert_eq!(got.view().bytes(), [0x7f, 1, 2, 0xff, 6, 3, 4, 5, 7, 8]);
    }

    #[test]
    fn generic_value_gather_preserves_raw_bytes() {
        let builder = Builder::new();
        let x = builder.constant("raw", TensorType::new(vec![3, 5], DType::E4M3FN));
        let index = builder.constant("index", TensorType::f32(vec![]));
        let y = builder.gather(x, 0, index);
        let graph = builder.finish(y);
        let bytes = vec![
            0x00, 0x80, 0x01, 0x7e, 0x7f, 0xff, 0x38, 0xb8, 0x55, 0xd5, 0xff, 0x7f, 0x01, 0x02,
            0x03,
        ];
        let input = e4m3fn_tensor(vec![3, 5], bytes).unwrap();
        let inputs = HashMap::from([
            (x.id, crate::Value::Host(input)),
            (
                index.id,
                crate::Value::Host(poot_tensor::HostTensor::f32(vec![], vec![1.0])),
            ),
        ]);

        let got = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap();
        let crate::Value::Host(got) = got else {
            panic!("Gather must retain e4m3fn storage")
        };
        assert_eq!(got.shape(), [5]);
        assert_eq!(got.view().bytes(), [0xff, 0x38, 0xb8, 0x55, 0xd5]);
    }

    /// Card 555: [`crate::Value::gather`] (the standalone entry point outside the
    /// walk's own equation loop) routes an I32 index through the exact `IndexValue` path - correct
    /// above 2^24, where the F32 index has already collided with its rounded neighbour.
    ///
    /// Red mutation (recorded): revert `Value::gather` to widen every index to f32. An I32 index
    /// above 2^24 then silently selects the rounded neighbor's row instead of the one requested -
    /// confirmed red, reverted, confirmed green.
    #[test]
    fn value_gather_routes_authoritative_i32_index_through_exact_path() {
        const COLLIDING_ID: i32 = 16_777_217;
        const ROUNDED_ID: i32 = 16_777_216;
        let axis_len = COLLIDING_ID as usize + 1;
        let mut bytes = vec![0u8; axis_len];
        bytes[ROUNDED_ID as usize] = 0x38;
        bytes[COLLIDING_ID as usize] = 0x40;
        let table = e4m3fn_tensor(vec![axis_len], bytes).unwrap();
        let value = crate::Value::Host(table);

        // An F32 index keeps the f32 path and its collision.
        let f32_only_index = poot_tensor::HostTensor::f32(vec![], vec![COLLIDING_ID as f32]);
        let crate::Value::Host(via_f32) = value.gather(0, &f32_only_index).unwrap() else {
            panic!("must stay e4m3fn")
        };
        assert_eq!(
            via_f32.view().bytes(),
            [0x38],
            "an F32 index must keep the existing f32 behaviour, collision included"
        );

        // An I32 index takes the exact path.
        let exact_index = poot_tensor::HostTensor::i32(vec![], vec![COLLIDING_ID]);
        let crate::Value::Host(via_exact) = value.gather(0, &exact_index).unwrap() else {
            panic!("must stay e4m3fn")
        };
        assert_eq!(via_exact.view().bytes(), [0x40]);
    }

    #[test]
    fn generic_value_dynamic_update_slice_supports_static_and_runtime_indices() {
        let cases = [(false, 1usize), (true, 2usize)];
        for (runtime, position) in cases {
            let builder = Builder::new();
            let operand = builder.constant("operand", TensorType::new(vec![2, 5], DType::E4M3FN));
            let update = builder.constant("update", TensorType::new(vec![2, 2], DType::E4M3FN));
            let index = runtime.then(|| builder.constant("index", TensorType::f32(vec![])));
            let output = match index {
                Some(index) => builder.dynamic_update_slice_dyn(operand, update, index, 1),
                None => builder.dynamic_update_slice(operand, update, position, 1),
            };
            let graph = builder.finish(output);
            let operand_bytes = vec![0x7f, 1, 2, 3, 0xff, 4, 5, 6, 7, 8];
            let update_bytes = vec![0xff, 0x7f, 0x55, 0xaa];
            let mut inputs = HashMap::from([
                (
                    operand.id,
                    crate::Value::Host(e4m3fn_tensor(vec![2, 5], operand_bytes.clone()).unwrap()),
                ),
                (
                    update.id,
                    crate::Value::Host(e4m3fn_tensor(vec![2, 2], update_bytes.clone()).unwrap()),
                ),
            ]);
            if let Some(index) = index {
                inputs.insert(
                    index.id,
                    crate::Value::Host(poot_tensor::HostTensor::f32(vec![], vec![position as f32])),
                );
            }

            let crate::Value::Host(got) = crate::eval(
                &graph,
                &inputs,
                crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
            )
            .map(|r| r.output)
            .unwrap() else {
                panic!("DynamicUpdateSlice must retain e4m3fn storage")
            };
            let mut expected = operand_bytes;
            for row in 0..2 {
                let output_start = row * 5 + position;
                let update_start = row * 2;
                expected[output_start..output_start + 2]
                    .copy_from_slice(&update_bytes[update_start..update_start + 2]);
            }
            assert_eq!(got.view().bytes(), expected, "runtime={runtime}");
            assert_eq!(
                PortableE4m3FnLayout::for_shape(got.shape()).device_bytes,
                16
            );
            assert!(got.view().bytes().contains(&0x7f));
            assert!(got.view().bytes().contains(&0xff));
        }
    }

    /// A dynamic index/scatter inverse declared F32 but bound as an I32 tensor (a payload that is
    /// not the declared dtype) is caught at bind time (`walk::preflight_bindings`), before any
    /// equation could index it; there is no implicit I32 -> F32 conversion at bind.
    #[test]
    fn generic_value_rejects_a_non_f32_index_binding_before_indexing() {
        let builder = Builder::new();
        let operand = builder.constant("operand", TensorType::new(vec![2, 5], DType::E4M3FN));
        let update = builder.constant("update", TensorType::new(vec![2, 2], DType::E4M3FN));
        let index = builder.constant("index", TensorType::f32(vec![]));
        let output = builder.dynamic_update_slice_dyn(operand, update, index, 1);
        let graph = builder.finish(output);
        let inputs = HashMap::from([
            (
                operand.id,
                crate::Value::Host(e4m3fn_tensor(vec![2, 5], vec![0; 10]).unwrap()),
            ),
            (
                update.id,
                crate::Value::Host(e4m3fn_tensor(vec![2, 2], vec![1; 4]).unwrap()),
            ),
            (
                index.id,
                crate::Value::Host(poot_tensor::HostTensor::i32(vec![], vec![1])),
            ),
        ]);

        let message = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap_err()
        .to_string();
        assert!(message.contains("different dtype"), "{message}");

        let builder = Builder::new();
        let base = builder.constant("base", TensorType::new(vec![2, 5], DType::E4M3FN));
        let src = builder.constant("src", TensorType::new(vec![1, 5], DType::E4M3FN));
        let inverse = builder.constant("inverse", TensorType::f32(vec![2]));
        let output = builder.scatter_update(base, src, inverse);
        let graph = builder.finish(output);
        let inputs = HashMap::from([
            (
                base.id,
                crate::Value::Host(e4m3fn_tensor(vec![2, 5], vec![0; 10]).unwrap()),
            ),
            (
                src.id,
                crate::Value::Host(e4m3fn_tensor(vec![1, 5], vec![1; 5]).unwrap()),
            ),
            (
                inverse.id,
                crate::Value::Host(poot_tensor::HostTensor::i32(vec![2], vec![0, -1])),
            ),
        ]);

        let message = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap_err()
        .to_string();
        assert!(message.contains("different dtype"), "{message}");
    }

    #[test]
    fn generic_value_broadcast_preserves_raw_bytes() {
        let builder = Builder::new();
        let x = builder.constant("raw", TensorType::new(vec![1, 3], DType::E4M3FN));
        let y = builder.broadcast(x, vec![2, 3]);
        let graph = builder.finish(y);
        let input = e4m3fn_tensor(vec![1, 3], vec![0x7f, 0xff, 0x01]).unwrap();
        let inputs = HashMap::from([(x.id, crate::Value::Host(input))]);

        let got = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap();
        let crate::Value::Host(got) = got else {
            panic!("Broadcast must retain e4m3fn storage")
        };
        assert_eq!(got.shape(), [2, 3]);
        assert_eq!(got.view().bytes(), [0x7f, 0xff, 0x01, 0x7f, 0xff, 0x01]);
    }

    #[test]
    fn generic_value_scatter_update_preserves_raw_bytes() {
        let builder = Builder::new();
        let base = builder.constant("base", TensorType::new(vec![3, 5], DType::E4M3FN));
        let src = builder.constant("src", TensorType::new(vec![2, 5], DType::E4M3FN));
        let inverse = builder.constant("inverse", TensorType::f32(vec![3]));
        let output = builder.scatter_update(base, src, inverse);
        let graph = builder.finish(output);
        let base_value = e4m3fn_tensor(
            vec![3, 5],
            vec![0x7f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        )
        .unwrap();
        let src_value =
            e4m3fn_tensor(vec![2, 5], vec![0xff, 21, 22, 23, 24, 0x7f, 31, 32, 33, 34]).unwrap();
        let inputs = HashMap::from([
            (base.id, crate::Value::Host(base_value)),
            (src.id, crate::Value::Host(src_value)),
            (
                inverse.id,
                crate::Value::Host(poot_tensor::HostTensor::f32(vec![3], vec![1.0, -1.0, 0.0])),
            ),
        ]);
        let crate::Value::Host(got) = crate::eval(
            &graph,
            &inputs,
            crate::EvalOptions::new(crate::EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output)
        .unwrap() else {
            panic!("ScatterUpdate must retain e4m3fn storage")
        };
        assert_eq!(
            got.view().bytes(),
            [0x7f, 31, 32, 33, 34, 5, 6, 7, 8, 9, 0xff, 21, 22, 23, 24]
        );
    }
}
