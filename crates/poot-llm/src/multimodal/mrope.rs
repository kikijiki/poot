//! Typed host binding for Card 152's packed mRoPE position slot.

use poot_tensor::DType;
use std::collections::HashMap;

use poot_eval::Value;
use poot_graph_ir::{Graph, Slot, ValueId};
use poot_models::mrope::{
    MAX_EXACT_MROPE_POSITION, MropePosition, MropePositionError, MropePositionIds,
};
use poot_tensor::HostTensor;

/// Per-request sectioned-RoPE cursor for production decode scheduling.
///
/// The cursor owns the authoritative prompt axes so prefix replay and recompute preemption can seek
/// back into the prompt without reconstructing rotary positions from the absolute KV position. Past
/// the prompt it advances collapsed generated-text positions from
/// [`MropePositionIds::next_text_position`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MropeDecodeState {
    positions: MropePositionIds,
    token_index: usize,
    current: MropePosition,
}

impl MropeDecodeState {
    /// Construct a cursor at `token_index`, validating the request's encoded prompt length first.
    pub fn new(
        positions: MropePositionIds,
        expected_prompt_len: usize,
        token_index: usize,
    ) -> Result<Self, MropePositionError> {
        if positions.len() != expected_prompt_len {
            return Err(MropePositionError::LengthMismatch {
                expected: expected_prompt_len,
                actual: positions.len(),
            });
        }
        let current = position_for_token_index(&positions, token_index)?;
        Ok(Self {
            positions,
            token_index,
            current,
        })
    }

    pub const fn current(&self) -> MropePosition {
        self.current
    }

    pub const fn token_index(&self) -> usize {
        self.token_index
    }

    pub fn prompt_len(&self) -> usize {
        self.positions.len()
    }

    /// Move to an explicit scheduler token index. Validation is atomic: overflow leaves the cursor
    /// unchanged.
    pub fn seek(&mut self, token_index: usize) -> Result<(), MropePositionError> {
        let current = position_for_token_index(&self.positions, token_index)?;
        self.token_index = token_index;
        self.current = current;
        Ok(())
    }

    /// Advance exactly one consumed row token. A scheduler holds a row by not calling this method.
    pub fn advance(&mut self) -> Result<(), MropePositionError> {
        let next = self
            .token_index
            .checked_add(1)
            .ok_or(MropePositionError::TokenCountOverflow)?;
        self.seek(next)
    }
}

fn position_for_token_index(
    positions: &MropePositionIds,
    token_index: usize,
) -> Result<MropePosition, MropePositionError> {
    if token_index < positions.len() {
        return positions.position(token_index);
    }
    let generated = token_index - positions.len();
    let base = positions.next_text_position().temporal;
    let generated = i32::try_from(generated).map_err(|_| MropePositionError::PositionOverflow {
        position: usize::MAX,
    })?;
    let position = base
        .checked_add(generated)
        .ok_or(MropePositionError::PositionOverflow {
            position: usize::MAX,
        })?;
    let position_usize =
        usize::try_from(position).map_err(|_| MropePositionError::PositionOverflow {
            position: usize::MAX,
        })?;
    if position_usize > MAX_EXACT_MROPE_POSITION {
        return Err(MropePositionError::PositionOverflow {
            position: position_usize,
        });
    }
    Ok(MropePosition::collapsed(position))
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MropeBindError {
    #[error("mRoPE graph has no Slot::MropePosition input")]
    MissingSlot,
    #[error(
        "mRoPE graph has {count} Slot::MropePosition inputs; exactly one packed slot is required"
    )]
    MultipleSlots { count: usize },
    #[error("mRoPE {mode} slot has shape {actual:?}; expected {expected:?}")]
    Shape {
        mode: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("mRoPE slot has dtype {actual}; expected i32")]
    WrongDType { actual: DType },
    #[error(transparent)]
    Position(#[from] MropePositionError),
}

/// Bind the one axis-major `[3,L]` mRoPE prefill slot into an input map.
pub fn bind_mrope_prefill_positions(
    graph: &Graph,
    inputs: &mut HashMap<ValueId, Value>,
    positions: &MropePositionIds,
) -> Result<(), MropeBindError> {
    let id = one_mrope_slot(graph)?;
    let expected = vec![3, positions.len()];
    let actual = graph.aval(id).shape.clone();
    if actual != expected {
        // Validate against the graph length, not the host object's own length, so callers receive the
        // precise typed length mismatch promised by the host position contract.
        if actual.len() == 2 && actual[0] == 3 {
            positions.packed_for_len(actual[1])?;
        }
        return Err(MropeBindError::Shape {
            mode: "prefill",
            expected,
            actual,
        });
    }
    inputs.insert(id, HostTensor::i32(actual, positions.packed()).into());
    Ok(())
}

/// Bind one temporal/height/width tuple into a scalar-decode `[3]` mRoPE slot.
pub fn bind_mrope_decode_position(
    graph: &Graph,
    inputs: &mut HashMap<ValueId, Value>,
    position: MropePosition,
) -> Result<(), MropeBindError> {
    let id = one_mrope_slot(graph)?;
    let expected = vec![3];
    let actual = graph.aval(id).shape.clone();
    if actual != expected {
        return Err(MropeBindError::Shape {
            mode: "decode",
            expected,
            actual,
        });
    }
    inputs.insert(
        id,
        HostTensor::i32(actual, position.packed().to_vec()).into(),
    );
    Ok(())
}

/// Bind one temporal/height/width tuple per row into an axis-major `[3,B]` batched-decode slot.
/// The complete slot contract is checked before `inputs` is mutated.
#[cfg(test)]
pub(crate) fn bind_mrope_batched_decode_positions(
    graph: &Graph,
    inputs: &mut HashMap<ValueId, Value>,
    positions: &[MropePosition],
) -> Result<(), MropeBindError> {
    let id = one_mrope_slot(graph)?;
    let expected = vec![3, positions.len()];
    let actual = graph.aval(id).shape.clone();
    if actual != expected {
        return Err(MropeBindError::Shape {
            mode: "batched decode",
            expected,
            actual,
        });
    }

    let mut packed = Vec::with_capacity(positions.len() * 3);
    packed.extend(positions.iter().map(|position| position.temporal));
    packed.extend(positions.iter().map(|position| position.height));
    packed.extend(positions.iter().map(|position| position.width));
    inputs.insert(id, HostTensor::i32(actual, packed).into());
    Ok(())
}

fn one_mrope_slot(graph: &Graph) -> Result<ValueId, MropeBindError> {
    let ids: Vec<ValueId> = graph
        .slots
        .iter()
        .filter_map(|&(id, slot)| (slot == Slot::MropePosition).then_some(id))
        .collect();
    match ids.as_slice() {
        [] => Err(MropeBindError::MissingSlot),
        [id] => {
            let actual = graph.aval(*id).dtype;
            if actual != DType::I32 {
                return Err(MropeBindError::WrongDType { actual });
            }
            Ok(*id)
        }
        _ => Err(MropeBindError::MultipleSlots { count: ids.len() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_graph_ir::{Builder, TensorType};
    use poot_models::mrope::{MropeSegment, build_mrope_position_ids};
    use poot_models::qwen2::{
        Qwen2Config, trace_decode_kv_masked_batched, trace_decode_mrope, trace_prefill_mrope,
    };

    fn tiny_cfg() -> Qwen2Config {
        Qwen2Config {
            vocab: 8,
            hidden: 8,
            inter: 16,
            layers: 0,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 8,
            rotary_dim: 8,
            max_pos: 32,
            qkv_bias: true,
            qk_norm: false,
            mrope_section: Some([1, 1, 2]),
            ..Default::default()
        }
    }

    #[test]
    fn mrope_prefill_binder_uses_one_axis_major_i32_slot() {
        let positions = build_mrope_position_ids(&[MropeSegment::Text(3)], 1).unwrap();
        let graph = trace_prefill_mrope(tiny_cfg(), 3);
        let mut inputs = HashMap::new();
        bind_mrope_prefill_positions(&graph, &mut inputs, &positions).unwrap();

        let slots: Vec<_> = graph
            .slots
            .iter()
            .filter(|(_, slot)| *slot == Slot::MropePosition)
            .collect();
        assert_eq!(slots.len(), 1);
        let tensor = &inputs[&slots[0].0];
        assert_eq!(tensor.as_host().expect("dense weight").shape(), vec![3, 3]);
        assert_eq!(
            tensor.as_host().expect("dense weight").as_i32(),
            Some(&[0, 1, 2, 0, 1, 2, 0, 1, 2][..])
        );
    }

    #[test]
    fn mrope_decode_binder_accepts_next_text_position() {
        let positions = build_mrope_position_ids(&[MropeSegment::Text(3)], 1).unwrap();
        let graph = trace_decode_mrope(tiny_cfg(), 3);
        let mut inputs = HashMap::new();
        bind_mrope_decode_position(&graph, &mut inputs, positions.next_text_position()).unwrap();
        let id = graph
            .slots
            .iter()
            .find_map(|&(id, slot)| (slot == Slot::MropePosition).then_some(id))
            .unwrap();
        assert_eq!(
            inputs[&id].as_host().expect("dense weight").as_i32(),
            Some(&[3, 3, 3][..])
        );
    }

    #[test]
    fn mrope_binder_rejects_length_shape_and_slot_contract_violations() {
        let positions = build_mrope_position_ids(&[MropeSegment::Text(2)], 1).unwrap();
        let graph = trace_prefill_mrope(tiny_cfg(), 3);
        assert_eq!(
            bind_mrope_prefill_positions(&graph, &mut HashMap::new(), &positions),
            Err(MropeBindError::Position(
                MropePositionError::LengthMismatch {
                    expected: 3,
                    actual: 2,
                }
            ))
        );

        let b = Builder::new();
        let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
        let no_mrope = b.finish(token);
        assert_eq!(
            bind_mrope_decode_position(&no_mrope, &mut HashMap::new(), MropePosition::collapsed(0)),
            Err(MropeBindError::MissingSlot)
        );
    }

    #[test]
    fn mrope_binders_reject_non_i32_slots_before_inserting_inputs() {
        let positions = build_mrope_position_ids(&[MropeSegment::Text(2)], 1).unwrap();

        let b = Builder::new();
        let slot = b.slot(Slot::MropePosition, TensorType::f32(vec![3, 2]));
        let prefill = b.finish(slot);
        let mut prefill_inputs = HashMap::new();
        assert_eq!(
            bind_mrope_prefill_positions(&prefill, &mut prefill_inputs, &positions),
            Err(MropeBindError::WrongDType { actual: DType::F32 })
        );
        assert!(prefill_inputs.is_empty());

        let b = Builder::new();
        let slot = b.slot(Slot::MropePosition, TensorType::f32(vec![3]));
        let decode = b.finish(slot);
        let mut decode_inputs = HashMap::new();
        assert_eq!(
            bind_mrope_decode_position(&decode, &mut decode_inputs, positions.next_text_position()),
            Err(MropeBindError::WrongDType { actual: DType::F32 })
        );
        assert!(decode_inputs.is_empty());
    }

    #[test]
    fn mrope_batched_decode_binder_packs_axis_major() {
        let graph = trace_decode_kv_masked_batched(tiny_cfg(), 8, 2);
        let positions = [
            MropePosition {
                temporal: 2,
                height: 5,
                width: 7,
            },
            MropePosition {
                temporal: 3,
                height: 4,
                width: 6,
            },
        ];
        let mut inputs = HashMap::new();
        bind_mrope_batched_decode_positions(&graph, &mut inputs, &positions).unwrap();
        let id = one_mrope_slot(&graph).unwrap();
        assert_eq!(
            inputs[&id].as_host().expect("dense weight").shape(),
            vec![3, 2]
        );
        assert_eq!(
            inputs[&id].as_host().expect("dense weight").as_i32(),
            Some(&[2, 3, 5, 4, 7, 6][..])
        );
    }

    #[test]
    fn mrope_batched_decode_binder_rejects_atomically() {
        let positions = [MropePosition::collapsed(1), MropePosition::collapsed(2)];
        let sentinel = HashMap::from([(999usize, Value::from(HostTensor::scalar(42.0)))]);

        let b = Builder::new();
        let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
        let graph = b.finish(token);
        let mut inputs = sentinel.clone();
        assert_eq!(
            bind_mrope_batched_decode_positions(&graph, &mut inputs, &positions),
            Err(MropeBindError::MissingSlot)
        );
        assert_eq!(inputs, sentinel);

        let b = Builder::new();
        let first = b.slot(Slot::MropePosition, TensorType::new(vec![3, 2], DType::I32));
        let _second = b.slot(Slot::MropePosition, TensorType::new(vec![3, 2], DType::I32));
        let graph = b.finish(first);
        let mut inputs = sentinel.clone();
        assert_eq!(
            bind_mrope_batched_decode_positions(&graph, &mut inputs, &positions),
            Err(MropeBindError::MultipleSlots { count: 2 })
        );
        assert_eq!(inputs, sentinel);

        let b = Builder::new();
        let slot = b.slot(Slot::MropePosition, TensorType::f32(vec![3, 2]));
        let graph = b.finish(slot);
        let mut inputs = sentinel.clone();
        assert_eq!(
            bind_mrope_batched_decode_positions(&graph, &mut inputs, &positions),
            Err(MropeBindError::WrongDType { actual: DType::F32 })
        );
        assert_eq!(inputs, sentinel);

        let b = Builder::new();
        let slot = b.slot(Slot::MropePosition, TensorType::new(vec![2, 2], DType::I32));
        let graph = b.finish(slot);
        let mut inputs = sentinel.clone();
        assert_eq!(
            bind_mrope_batched_decode_positions(&graph, &mut inputs, &positions),
            Err(MropeBindError::Shape {
                mode: "batched decode",
                expected: vec![3, 2],
                actual: vec![2, 2],
            })
        );
        assert_eq!(inputs, sentinel);

        let graph = trace_decode_kv_masked_batched(tiny_cfg(), 8, 2);
        let mut inputs = sentinel.clone();
        assert_eq!(
            bind_mrope_batched_decode_positions(&graph, &mut inputs, &positions[..1]),
            Err(MropeBindError::Shape {
                mode: "batched decode",
                expected: vec![3, 1],
                actual: vec![3, 2],
            })
        );
        assert_eq!(inputs, sentinel);
    }

    #[test]
    fn mrope_decode_state_seeks_prompt_and_advances_generated_text() {
        let positions = build_mrope_position_ids(
            &[
                MropeSegment::Text(1),
                MropeSegment::Image(poot_models::mrope::MropeGrid::new(1, 2, 4)),
                MropeSegment::Text(1),
            ],
            2,
        )
        .unwrap();
        assert_eq!(positions.len(), 4);

        let mut state = MropeDecodeState::new(positions.clone(), 4, 1).unwrap();
        assert_eq!(state.current(), positions.position(1).unwrap());
        state.seek(3).unwrap();
        assert_eq!(state.current(), positions.position(3).unwrap());
        state.advance().unwrap();
        assert_eq!(state.token_index(), 4);
        assert_eq!(state.current(), positions.next_text_position());
        state.advance().unwrap();
        let next = positions.next_text_position().temporal + 1;
        assert_eq!(state.current(), MropePosition::collapsed(next));

        let held = state.clone();
        let mut advancing = state;
        advancing.advance().unwrap();
        assert_eq!(held.token_index(), 5, "a held row keeps its cursor");
        assert_eq!(advancing.token_index(), 6);
    }

    #[test]
    fn mrope_decode_state_rejects_prompt_length_mismatch() {
        let positions = build_mrope_position_ids(&[MropeSegment::Text(2)], 1).unwrap();
        assert_eq!(
            MropeDecodeState::new(positions, 3, 0),
            Err(MropePositionError::LengthMismatch {
                expected: 3,
                actual: 2,
            })
        );
    }
}
