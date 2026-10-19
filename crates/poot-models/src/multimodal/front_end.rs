//! The front end of a vision-language family: a conversation with images in, a prompt and its request
//! inputs out (ADR-0117 d6).
//!
//! A family's registry entry builds a [`FrontEnd`] beside its model ([`ModelParts`](crate::registry::ModelParts)).
//! The front end preprocesses the images, runs the family's vision towers through [`Towers`] (which the
//! driver crate implements with one driver per tower), renders the conversation and returns what any text
//! driver takes: the prompt's token ids and the values of the request inputs its model declares (the
//! embedding rows at the image placeholders and, for a model that rotates at sectioned positions, the
//! prompt's position table). The driver crate names no family: it decodes the bytes into an [`Image`] and
//! asks the model's front end.

use std::error::Error;
use std::fmt;

use poot_text::TextCodec;

use super::mrope::MropePositionIds;
use crate::components::embed::embeds_inputs;
use crate::components::rope::MROPE;
use crate::io::{Fill, InputValue, RequestInputs, TokenValues, ValueError};
use poot_tensor::HostTensor;

/// A vision tower of a model container, named by its family: the key a front end runs it under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TowerId(&'static str);

impl TowerId {
    /// The tower named `name`.
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// The name, for messages and receipts.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for TowerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// Why pixels are not an [`Image`].
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum ImageError {
    /// A dimension is zero.
    #[error("image dimensions must be nonzero, got {height}x{width}")]
    ZeroDimension {
        /// Rows of pixels.
        height: usize,
        /// Pixels per row.
        width: usize,
    },
    /// The pixel buffer is not `height * width * 3` bytes.
    #[error("a {height}x{width} RGB8 image holds {expected} bytes, got {actual}")]
    PixelDataLen {
        /// Rows of pixels.
        height: usize,
        /// Pixels per row.
        width: usize,
        /// The byte count the dimensions require.
        expected: usize,
        /// The byte count given.
        actual: usize,
    },
}

/// An image as row-major interleaved RGB8, `height * width * 3` bytes, both dimensions nonzero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    height: usize,
    width: usize,
    rgb: Vec<u8>,
}

impl Image {
    /// An image of owned row-major, interleaved RGB8 pixels.
    pub fn from_rgb8(height: usize, width: usize, rgb: Vec<u8>) -> Result<Self, ImageError> {
        if height == 0 || width == 0 {
            return Err(ImageError::ZeroDimension { height, width });
        }
        let expected = height.checked_mul(width).and_then(|n| n.checked_mul(3));
        if expected != Some(rgb.len()) {
            return Err(ImageError::PixelDataLen {
                height,
                width,
                expected: expected.unwrap_or(usize::MAX),
                actual: rgb.len(),
            });
        }
        Ok(Self { height, width, rgb })
    }

    /// Rows of pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Pixels per row.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The pixels, row-major, three bytes (R, G, B) each.
    pub fn rgb(&self) -> &[u8] {
        &self.rgb
    }
}

/// What a clip says about itself before any frame is decoded: the facts frame sampling needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoMetadata<'a> {
    /// The clip's length in seconds; finite and positive, and no timestamp exceeds it.
    pub duration_seconds: f64,
    /// The source's frame rate in frames per second; finite and positive.
    pub source_frames_per_second: f64,
    /// The number of source frames; at least one, and equal to `frame_timestamps_seconds.len()`.
    pub frame_count: usize,
    /// One presentation timestamp in seconds per source frame: finite, nonnegative and strictly
    /// increasing.
    pub frame_timestamps_seconds: &'a [f64],
}

/// The caller's container decoder: decodes only the frames a front end selects.
pub trait FrameSource {
    /// Decode source frame `index`. Callers ask for strictly increasing indices.
    fn frame(&mut self, index: usize) -> Result<Image, Box<dyn Error + Send + Sync>>;
}

/// A probed clip and the decoder of its frames.
pub struct Video<'a> {
    /// What the clip says about itself.
    pub metadata: VideoMetadata<'a>,
    /// Decodes the frames the front end selects.
    pub frames: &'a mut dyn FrameSource,
}

/// One image or video of a request.
pub enum Media<'a> {
    /// A decoded still image.
    Image(&'a Image),
    /// A clip whose frames are decoded on demand.
    Video(Video<'a>),
}

/// One turn of a conversation that holds media: the role, how many media items the turn carries, and
/// its text. The turn's media come first in its content, then the text (the order a served request
/// flattens to). The media themselves are passed beside the turns, in the order of the turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChatTurn<'a> {
    /// The turn's role as the chat template names it (`"user"`, `"assistant"`, `"system"`).
    pub role: &'a str,
    /// How many of the request's media items this turn carries; they lead the turn's content.
    pub media: usize,
    /// The turn's text, after its media.
    pub text: &'a str,
}

impl<'a> ChatTurn<'a> {
    /// A user turn of `media` items and the question about them.
    pub fn user(media: usize, text: &'a str) -> Self {
        Self {
            role: "user",
            media,
            text,
        }
    }
}

/// A request with media lowered to what the text driver takes.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// The prompt's token ids, the media placeholders included.
    pub prompt: Vec<u32>,
    /// The values for the inputs the text model declares: the embedding rows that replace the
    /// placeholders and, for a model with sectioned positions, where every prompt token rotates.
    pub inputs: RequestInputs,
}

/// Why a tower could not encode a row.
#[derive(Debug, thiserror::Error)]
pub enum TowerError {
    /// The row is the client's to fix: more patches than the tower takes.
    #[error("{0}")]
    Refused(String),
    /// The tower or its driver failed.
    #[error(transparent)]
    Failed(Box<dyn Error + Send + Sync>),
}

/// The towers of a model container, as a front end runs them. The driver crate implements it with one
/// driver per tower.
pub trait Towers {
    /// Encode one row, whose values for the tower's request inputs are `inputs`: the row's embedding
    /// rows, `positions * width` values row-major.
    fn encode(&mut self, tower: TowerId, inputs: &RequestInputs) -> Result<Vec<f32>, TowerError>;
}

/// Why a conversation with media did not become a prompt.
#[derive(Debug, thiserror::Error)]
pub enum FrontEndError {
    /// The media or the question are the client's to fix.
    #[error("{0}")]
    Refused(String),
    /// The front end or its tower failed.
    #[error(transparent)]
    Failed(Box<dyn Error + Send + Sync>),
}

impl From<TowerError> for FrontEndError {
    fn from(error: TowerError) -> Self {
        match error {
            TowerError::Refused(reason) => Self::Refused(reason),
            failed @ TowerError::Failed(_) => Self::Failed(Box::new(failed)),
        }
    }
}

/// The media side of a vision-language family: media and a conversation in, a prompt out. It never
/// generates, and it is shared by every request.
pub trait FrontEnd: Send + Sync + fmt::Debug {
    /// Preprocess `media` (in the order of `turns`), encode them on `towers`, render the conversation
    /// `turns` through `text`'s chat template and assemble the prompt and its request inputs.
    fn prepare(
        &self,
        turns: &[ChatTurn<'_>],
        media: &mut [Media<'_>],
        towers: &mut dyn Towers,
        text: &TextCodec,
    ) -> Result<Prepared, FrontEndError>;
}

/// The request inputs of a multimodal prompt: `rows` (`positions.len() * width` values, row-major) at
/// the prompt's `positions`, and for a model with sectioned positions the prompt's position table.
pub fn prompt_inputs(
    width: usize,
    positions: Vec<usize>,
    rows: Vec<f32>,
    mrope: Option<&MropePositionIds>,
) -> Result<RequestInputs, ValueError> {
    let inputs = embeds_inputs(width, positions, rows)?;
    Ok(match mrope {
        Some(ids) => inputs.with(MROPE, InputValue::Tokens(mrope_values(ids)?)),
        None => inputs,
    })
}

/// A prompt's sectioned positions as dense three-axis values over the prompt, continuing past it from
/// the collapsed position after the last token, one further per generated token.
fn mrope_values(ids: &MropePositionIds) -> Result<TokenValues, ValueError> {
    let [temporal, height, width] = ids.axes();
    let data: Vec<i32> = (0..ids.len())
        .flat_map(|i| [temporal[i], height[i], width[i]])
        .collect();
    let next = ids.next_text_position().temporal;
    Ok(
        TokenValues::dense(HostTensor::i32(vec![ids.len(), 3], data))?.with_beyond(Fill::Advance {
            from: vec![next; 3],
            step: 1,
        }),
    )
}

/// Why a conversation's turns do not carry the media a request holds.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum TurnMediaError {
    /// The request holds no media: a front end serves media requests only.
    #[error("a request needs at least one image")]
    NoMedia,
    /// The turns count other media than the request holds.
    #[error("the turns hold {turns} media items but the request has {media}")]
    Mismatch {
        /// The media count the turns declare.
        turns: usize,
        /// The media count the request holds.
        media: usize,
    },
}

impl From<TurnMediaError> for FrontEndError {
    fn from(error: TurnMediaError) -> Self {
        Self::Refused(error.to_string())
    }
}

/// Whether `turns` carry exactly the `media` items a request holds.
pub fn check_turn_media(turns: &[ChatTurn<'_>], media: usize) -> Result<(), TurnMediaError> {
    if media == 0 {
        return Err(TurnMediaError::NoMedia);
    }
    let in_turns: usize = turns.iter().map(|turn| turn.media).sum();
    if in_turns != media {
        return Err(TurnMediaError::Mismatch {
            turns: in_turns,
            media,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::mrope::{MropeGrid, MropeSegment, build_mrope_position_ids};
    use crate::components::embed::EMBEDS;

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

    /// A prompt's sectioned positions become dense three-axis values over the prompt: the image's four
    /// tokens sit on a 2 x 2 grid at one temporal position and the text after it continues from the
    /// largest position plus one, and past the prompt the values advance by one per position from that
    /// collapsed continuation. The binder (`driver::bind`) evaluates exactly this data, so the table here
    /// is the position every token of the prompt rotates at. Mutation: build the data in token-major
    /// instead of per-token `[temporal, height, width]` order, or start the continuation one early; the
    /// table below goes red.
    #[test]
    fn a_prompts_sectioned_positions_become_dense_values_with_a_continuation() {
        let values = mrope_values(&ids()).unwrap();
        assert_eq!(values.support().len(), 8);
        assert_eq!(values.width(), 3);
        let words = values.data().as_i32().unwrap();
        let at = |i: usize| [words[3 * i], words[3 * i + 1], words[3 * i + 2]];
        assert_eq!([at(0), at(1)], [[0, 0, 0], [1, 1, 1]]);
        assert_eq!(
            [at(2), at(3), at(4), at(5)],
            [[2, 2, 2], [2, 2, 3], [2, 3, 2], [2, 3, 3]]
        );
        assert_eq!([at(6), at(7)], [[4, 4, 4], [5, 5, 5]]);
        assert_eq!(
            values.beyond(),
            &Fill::Advance {
                from: vec![6; 3],
                step: 1
            }
        );
    }

    /// The embedding rows of a prompt are sparse values at its placeholders, and the prompt inputs hold
    /// both the rows and (for a model with sectioned positions) the position table.
    #[test]
    fn prompt_inputs_hold_the_rows_and_the_positions_a_model_declares() {
        let rows = vec![1.0, 2.0, 3.0, 4.0];
        let plain = prompt_inputs(2, vec![1, 3], rows.clone(), None).unwrap();
        assert!(plain.get(EMBEDS).is_some() && plain.get(MROPE).is_none());
        let sectioned = prompt_inputs(2, vec![1, 3], rows, Some(&ids())).unwrap();
        assert!(sectioned.get(EMBEDS).is_some() && sectioned.get(MROPE).is_some());
        let Some(InputValue::Tokens(embeds)) = sectioned.get(EMBEDS) else {
            panic!("the rows are per-token values");
        };
        assert_eq!(embeds.support(), &crate::io::Support::Sparse(vec![1, 3]));
        assert!(prompt_inputs(2, vec![3, 1], vec![0.0; 4], None).is_err());
    }

    /// A request whose turns count other media than it holds, or holds none, is refused.
    #[test]
    fn turns_must_carry_exactly_the_requests_media() {
        let turns = [ChatTurn::user(2, "what")];
        assert!(check_turn_media(&turns, 2).is_ok());
        assert_eq!(
            check_turn_media(&turns, 1),
            Err(TurnMediaError::Mismatch { turns: 2, media: 1 })
        );
        assert_eq!(
            check_turn_media(&[ChatTurn::user(0, "what")], 0),
            Err(TurnMediaError::NoMedia)
        );
    }

    /// An image whose pixel buffer does not match its dimensions, or whose dimension is zero, is refused.
    #[test]
    fn an_image_checks_its_dimensions_against_its_pixels() {
        assert!(Image::from_rgb8(1, 2, vec![0; 6]).is_ok());
        assert_eq!(
            Image::from_rgb8(1, 2, vec![0; 5]),
            Err(ImageError::PixelDataLen {
                height: 1,
                width: 2,
                expected: 6,
                actual: 5
            })
        );
        assert!(matches!(
            Image::from_rgb8(0, 2, Vec::new()),
            Err(ImageError::ZeroDimension { .. })
        ));
    }
}
