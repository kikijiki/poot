//! The sectioned-RoPE position cursor of a Qwen-VL decode (held for POOT-650).

use poot_models::mrope::{
    MAX_EXACT_MROPE_POSITION, MropePosition, MropePositionError, MropePositionIds,
};

/// Per-request sectioned-RoPE cursor for production decode scheduling.
///
/// The cursor owns the authoritative prompt axes so prefix replay and recompute preemption can seek
/// back into the prompt without reconstructing rotary positions from the absolute KV position. Past
/// the prompt it advances collapsed generated-text positions from
/// [`MropePositionIds::next_text_position`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), expect(dead_code, reason = "held for POOT-650"))]
pub(crate) struct MropeDecodeState {
    positions: MropePositionIds,
    token_index: usize,
    current: MropePosition,
}

#[cfg_attr(not(test), expect(dead_code, reason = "held for POOT-650"))]
impl MropeDecodeState {
    /// Construct a cursor at `token_index`, validating the request's encoded prompt length first.
    pub(crate) fn new(
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

    pub(crate) const fn current(&self) -> MropePosition {
        self.current
    }

    pub(crate) const fn token_index(&self) -> usize {
        self.token_index
    }

    /// Move to an explicit scheduler token index. Validation is atomic: overflow leaves the cursor
    /// unchanged.
    pub(crate) fn seek(&mut self, token_index: usize) -> Result<(), MropePositionError> {
        let current = position_for_token_index(&self.positions, token_index)?;
        self.token_index = token_index;
        self.current = current;
        Ok(())
    }

    /// Advance exactly one consumed row token. A scheduler holds a row by not calling this method.
    pub(crate) fn advance(&mut self) -> Result<(), MropePositionError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use poot_models::mrope::{MropeSegment, build_mrope_position_ids};

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
