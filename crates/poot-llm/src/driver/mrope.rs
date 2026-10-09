//! The sectioned rotary positions of a multimodal prompt (`Slot::MropePosition`).
//!
//! A Qwen2.5-VL prompt's tokens do not all rotate at their index: the text before an image sits at its
//! index, the image's tokens at a temporal, a height and a width position of their own, and the text
//! after it continues from the largest of those plus one. The processor's [`MropePositionIds`] holds
//! the prompt's three axes and that continuation; a sequence opened with them
//! ([`Driver::open_with_mrope`](super::Driver::open_with_mrope)) binds, at every step, the position its
//! token has: the prompt's own axes inside the prompt, and the collapsed continuation after it, one
//! further per generated token.

use poot_models::mrope::MropePositionIds;

/// The `[temporal, height, width]` rotary position of the token at sequence index `index` of a prompt
/// whose positions are `ids`: its own axes inside the prompt, then one more than the previous token on
/// every axis from [`MropePositionIds::next_text_position`] on.
pub(super) fn rotary_position(ids: &MropePositionIds, index: usize) -> [i32; 3] {
    let prompt = ids.len();
    if index < prompt {
        let [temporal, height, width] = ids.axes();
        return [temporal[index], height[index], width[index]];
    }
    let next = ids.next_text_position().temporal;
    // Capacity bounds `index`, and positions past `i32` are not exact in the oracle either.
    let position = next + i32::try_from(index - prompt).expect("a generated index fits an i32");
    [position; 3]
}

#[cfg(test)]
mod tests {
    use poot_models::mrope::{MropeGrid, MropeSegment, build_mrope_position_ids};

    use super::*;

    /// Two text tokens, one 1 x 4 x 4 image (a 2 x 2 block of merged tokens), two text tokens.
    fn ids() -> MropePositionIds {
        build_mrope_position_ids(
            &[
                MropeSegment::Text(2),
                MropeSegment::Image(MropeGrid::new(1, 4, 4)),
                MropeSegment::Text(2),
            ],
            2,
        )
        .unwrap()
    }

    /// The image's four tokens sit on a 2 x 2 grid at one temporal position, the text after it
    /// continues from the largest position plus one, and tokens past the prompt step one at a time.
    #[test]
    fn positions_are_the_prompts_axes_then_the_collapsed_continuation() {
        let ids = ids();
        let at = |i| rotary_position(&ids, i);
        assert_eq!([at(0), at(1)], [[0, 0, 0], [1, 1, 1]]);
        assert_eq!(
            [at(2), at(3), at(4), at(5)],
            [[2, 2, 2], [2, 2, 3], [2, 3, 2], [2, 3, 3]]
        );
        assert_eq!([at(6), at(7)], [[4, 4, 4], [5, 5, 5]]);
        assert_eq!([at(8), at(9)], [[6, 6, 6], [7, 7, 7]]);
    }
}
