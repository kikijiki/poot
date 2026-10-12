//! [`PromptEmbeds`]: the embedding rows a prompt supplies in place of its tokens' own (a multimodal
//! prompt's image rows at its `<image>` placeholders). They are step inputs (`StepInput::TokensWithEmbeds`),
//! bound by value on every prefill step that covers one of their positions: nothing about them is
//! baked into a program, so two prompts of the same shape and different images share every entry.

use super::error::EmbedsError;

/// Embedding rows at some positions of a prompt: `rows` holds one `width`-wide row per entry of
/// `positions`, which are strictly ascending prompt positions.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptEmbeds {
    width: usize,
    positions: Vec<usize>,
    rows: Vec<f32>,
}

impl PromptEmbeds {
    /// Rows `rows` (row-major, `positions.len() * width` values) over `positions`.
    pub fn new(width: usize, positions: Vec<usize>, rows: Vec<f32>) -> Result<Self, EmbedsError> {
        if rows.len() != positions.len() * width {
            return Err(EmbedsError::Length {
                positions: positions.len(),
                width,
                rows: rows.len(),
            });
        }
        Ok(Self {
            width,
            positions,
            rows,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    /// The prompt positions the rows replace, ascending.
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    /// Check the rows against a prompt of `prompt` tokens: ascending and inside it.
    pub(super) fn check_prompt(&self, prompt: usize) -> Result<(), EmbedsError> {
        let mut previous = None;
        for &position in &self.positions {
            if position >= prompt || previous.is_some_and(|p| p >= position) {
                return Err(EmbedsError::Position { position, prompt });
            }
            previous = Some(position);
        }
        Ok(())
    }

    /// The row replacing `position`, if one does.
    pub(super) fn row_at(&self, position: usize) -> Option<&[f32]> {
        let index = self.positions.binary_search(&position).ok()?;
        Some(&self.rows[index * self.width..(index + 1) * self.width])
    }

    /// Whether any position of `start..end` has a row.
    pub(super) fn covers(&self, start: usize, end: usize) -> bool {
        let first = self.positions.partition_point(|&p| p < start);
        self.positions.get(first).is_some_and(|&p| p < end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_checked_against_their_positions_and_the_prompt() {
        assert_eq!(
            PromptEmbeds::new(2, vec![1, 3], vec![0.0; 3]),
            Err(EmbedsError::Length {
                positions: 2,
                width: 2,
                rows: 3
            })
        );
        let embeds = PromptEmbeds::new(2, vec![1, 3], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(embeds.check_prompt(4), Ok(()));
        assert_eq!(
            embeds.check_prompt(3),
            Err(EmbedsError::Position {
                position: 3,
                prompt: 3
            })
        );
        let unordered = PromptEmbeds::new(1, vec![2, 2], vec![0.0; 2]).unwrap();
        assert_eq!(
            unordered.check_prompt(8),
            Err(EmbedsError::Position {
                position: 2,
                prompt: 8
            })
        );
        assert_eq!(embeds.row_at(3), Some(&[3.0, 4.0][..]));
        assert_eq!(embeds.row_at(2), None);
        assert!(embeds.covers(0, 2) && embeds.covers(3, 9) && !embeds.covers(4, 9));
        assert!(!embeds.covers(2, 3));
    }

    impl PromptEmbeds {
        /// The rows of entries `entries` (indices into [`Self::positions`]), row-major.
        pub(in crate::driver) fn rows_for(&self, entries: std::ops::Range<usize>) -> Vec<f32> {
            self.rows[entries.start * self.width..entries.end * self.width].to_vec()
        }
    }
}
