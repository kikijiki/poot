//! [`Driver::encode`]: one `Encode` step over a batch of rows, returning each row's final hidden
//! states (or scores) for the host to pool or rank (POOT-739, dmodel `Phase::Encode`).
//!
//! The rows are either token sequences (an embedding or reranking encoder, or a decoder's embedding
//! pass), image frames (a vision tower of a fixed grid) or patch rows (a vision tower whose windows
//! depend on the image's grid). A step runs on one entry of its shape, prepared on first
//! use like `generate`'s and kept under the same limits; token rows are different: their entries are
//! admitted by [`Driver::prepare_encode`] before serving, at a fixed row count and every power-of-two
//! length up to the longest row, and a step runs on the smallest prepared length holding its longest
//! row (a padded position is masked, never attended to; a padded row is dropped from the answer). A
//! token request the prepared shapes cannot hold is a typed refusal, never a compile. An encode step
//! carries no cache any later step reads, so it refuses while a sequence is open (its cache would alias
//! the sequence's).

use std::num::NonZeroUsize;

use poot_executor::{ExecError, ExecutableId};
use poot_models::model::{KvLayout, LogitRows, Phase, StepShape, TraceError};
use poot_models::qwen_vl_vision::PatchGeometry;

use super::caps::DriverCaps;
use super::error::{DriverError, EncodeError, InvalidOptions, InvalidRequest, Unsupported};
use super::host_sample;
use super::suffix::Head;
use super::{Driver, EntryKey, ModelRole, SlotRole, StepData};

/// The token-row shapes a driver prepares for encoding: `rows` rows per step, every length a step
/// can pad to up to `max_len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeShapes {
    /// Rows of every prepared step; a request of fewer pads, a request of more is refused.
    pub rows: NonZeroUsize,
    /// The longest row, in tokens, a request may carry.
    pub max_len: NonZeroUsize,
}

impl EncodeShapes {
    /// The prepared lengths, ascending: every power of two below `max_len`, then `max_len` itself.
    pub fn lengths(self) -> Vec<NonZeroUsize> {
        let max = self.max_len.get();
        let mut lengths: Vec<NonZeroUsize> =
            std::iter::successors(Some(1usize), |len| len.checked_mul(2))
                .take_while(|&len| len < max)
                .map(|len| NonZeroUsize::new(len).expect("a power of two is nonzero"))
                .collect();
        lengths.push(self.max_len);
        lengths
    }

    pub(super) fn caps(self) -> EncodeCaps {
        EncodeCaps {
            rows: self.rows,
            lengths: self.lengths(),
        }
    }
}

/// What [`Driver::prepare_encode`] admitted, as the caps publish it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodeCaps {
    /// Rows of every prepared step.
    pub rows: NonZeroUsize,
    /// The prepared lengths, ascending; the last is the longest row a request may carry.
    pub lengths: Vec<NonZeroUsize>,
}

/// One row of an encode step.
#[derive(Clone, Copy, Debug)]
pub enum EncodeRow<'a> {
    /// A token sequence, with each token's segment id (`types` is empty for all-zero ids; a
    /// cross-encoder's second sentence carries ones).
    Tokens {
        /// Token ids of the row.
        ids: &'a [u32],
        /// Segment id of each token, or empty for all zeros.
        types: &'a [u32],
    },
    /// One image frame of a vision tower: its normalized pixels, channel first, and the positions
    /// (patches) the tower traces per frame.
    Frame {
        /// Normalized pixel values, channel first.
        pixels: &'a [f32],
        /// Positions (patches) the tower traces for the frame.
        positions: usize,
    },
    /// One image or video of a windowed vision tower: its patches in the tower's window order
    /// (`patch_dim` values each) and the grid facts the tower attends and rotates by.
    Patches {
        /// The patches' pixel values in the tower's window order.
        pixels: &'a [f32],
        /// Grid facts the tower attends and rotates by.
        geometry: &'a PatchGeometry,
    },
}

/// What one [`Driver::encode`] step runs: every row of one kind.
#[derive(Clone, Copy, Debug)]
pub struct EncodeRequest<'a> {
    /// The rows of the step; every row is one kind.
    pub rows: &'a [EncodeRow<'a>],
}

/// The result of one encode step: a vector of `width` values per output position of every row.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    /// The width of one output position.
    pub width: usize,
    /// Per row, `positions * width` values row-major: every real token of a token row (its padding
    /// dropped) for a hidden-state model, the single scored position of a classifier, every output
    /// position of a frame.
    pub rows: Vec<Vec<f32>>,
}

impl Driver {
    /// Turn a failed encode step into the driver's answer. The step graph's cache is scratch no later
    /// step reads, so a device failure that left the device usable (a fault, a failed transaction) is
    /// proven recoverable by resetting the executable's state: that is [`DriverError::EncodeStep`].
    /// A poisoned device, a failed reset or any other error is returned as it came.
    fn recover_encode(&mut self, exe: ExecutableId, error: DriverError) -> DriverError {
        let DriverError::Device(
            step @ (ExecError::Device(_) | ExecError::Fault { .. } | ExecError::NeedsReset(_)),
        ) = error
        else {
            return error;
        };
        match self.reset_exe(exe) {
            Ok(()) => DriverError::EncodeStep(step),
            Err(reset) => reset,
        }
    }

    /// Admit `shapes`: prepare the `Encode` entry of every length at `shapes.rows` rows, then publish the
    /// caps. This is the only place token-row entries are compiled; [`Driver::encode`] refuses a shape
    /// that is not prepared.
    pub fn prepare_encode(&mut self, shapes: &EncodeShapes) -> Result<&DriverCaps, DriverError> {
        if self.encode_shapes.is_some_and(|held| held != *shapes) {
            return Err(InvalidOptions::EncodeShapesChanged.into());
        }
        for len in shapes.lengths() {
            self.prepared(Self::token_key(shapes.rows, len))?;
        }
        self.encode_shapes = Some(*shapes);
        Ok(self.refresh_caps(self.layout))
    }

    /// [`Driver::prepare_encode`] for a model that may not encode at all (a generative family whose body
    /// has no embedding pass, or a placement with no form of an encode step): `Ok(false)` when the
    /// family or the placement refuses it, with nothing prepared; any other refusal is the error.
    pub fn prepare_encode_if_traced(&mut self, shapes: &EncodeShapes) -> Result<bool, DriverError> {
        match self.prepare_encode(shapes) {
            Ok(_) => Ok(true),
            Err(DriverError::Unsupported(
                Unsupported::Trace(TraceError::PhaseUnsupported { .. })
                | Unsupported::Placement { .. },
            )) => Ok(false),
            Err(other) => Err(other),
        }
    }

    fn token_key(rows: NonZeroUsize, tokens: NonZeroUsize) -> EntryKey {
        EntryKey {
            role: ModelRole::Target,
            phase: Phase::Encode,
            shape: StepShape {
                rows,
                tokens,
                capacity: tokens,
                kv: KvLayout::Contiguous,
                logits: LogitRows::All,
                hidden: None,
            },
            head: Head::Logits,
            adapters: None,
        }
    }

    /// The lazily prepared entry of a frame or patch step: its shape comes from the image, so it is
    /// prepared on first use. Not reachable from the engine thread (it only sends token rows), but it is
    /// reachable from a served image chat: the vision worker's own driver encodes frames after startup,
    /// outside the engine plan's fit. Making that a prepared, accounted shape is the driver input
    /// contract (POOT-1224).
    fn prepared_frames(
        &mut self,
        rows: usize,
        positions: usize,
    ) -> Result<(EntryKey, super::Prepared), DriverError> {
        let tokens = NonZeroUsize::new(positions).expect("a row has a position");
        let key = Self::token_key(NonZeroUsize::new(rows).expect("checked non-empty"), tokens);
        let entry = self.prepared(key)?;
        Ok((key, entry))
    }

    /// Run one encode step over `request`'s rows.
    pub fn encode(&mut self, request: &EncodeRequest<'_>) -> Result<Encoded, DriverError> {
        let EncodeRequest { rows } = *request;
        if rows.is_empty() {
            return Err(InvalidRequest::Encode(EncodeError::Empty).into());
        }
        if self
            .serving
            .as_ref()
            .is_some_and(|serving| !serving.seqs.is_empty())
        {
            return Err(InvalidRequest::Encode(EncodeError::SequenceOpen).into());
        }
        let n = rows.len();
        // The key's row count: a token step runs at the prepared rows (the rest pad), any other kind at
        // the request's own.
        let (key, entry) = match rows[0] {
            EncodeRow::Tokens { .. } => {
                let shapes = self
                    .encode_shapes
                    .ok_or(InvalidRequest::Encode(EncodeError::NotPrepared))?;
                if n > shapes.rows.get() {
                    return Err(InvalidRequest::Encode(EncodeError::TooManyRows {
                        rows: n,
                        max: shapes.rows.get(),
                    })
                    .into());
                }
                let mut longest = 0;
                for (row, entry) in rows.iter().enumerate() {
                    let EncodeRow::Tokens { ids, types } = *entry else {
                        return Err(InvalidRequest::Encode(EncodeError::Mixed).into());
                    };
                    if ids.is_empty() {
                        return Err(InvalidRequest::Encode(EncodeError::EmptyRow { row }).into());
                    }
                    if !types.is_empty() && types.len() != ids.len() {
                        return Err(InvalidRequest::Encode(EncodeError::TokenTypes {
                            row,
                            tokens: ids.len(),
                            types: types.len(),
                        })
                        .into());
                    }
                    if ids.len() > shapes.max_len.get() {
                        return Err(InvalidRequest::Encode(EncodeError::RowTooLong {
                            row,
                            tokens: ids.len(),
                            max: shapes.max_len.get(),
                        })
                        .into());
                    }
                    longest = longest.max(ids.len());
                }
                let len = shapes
                    .lengths()
                    .into_iter()
                    .find(|len| len.get() >= longest)
                    .expect("the longest length is max_len, which holds every checked row");
                let key = Self::token_key(shapes.rows, len);
                let entry = self
                    .entries
                    .get(&key)
                    .cloned()
                    .ok_or(InvalidRequest::Encode(EncodeError::NotPrepared))?;
                (key, entry)
            }
            EncodeRow::Patches { geometry, .. } => {
                let positions = geometry.patches();
                if rows.iter().any(
                    |row| !matches!(row, EncodeRow::Patches { geometry, .. } if geometry.patches() == positions),
                ) {
                    return Err(InvalidRequest::Encode(EncodeError::Mixed).into());
                }
                self.prepared_frames(n, positions)?
            }
            EncodeRow::Frame { positions, .. } => {
                if rows.iter().any(
                    |row| !matches!(row, EncodeRow::Frame { positions: p, .. } if *p == positions),
                ) {
                    return Err(InvalidRequest::Encode(EncodeError::Mixed).into());
                }
                self.prepared_frames(n, positions)?
            }
        };
        let batch = key.shape.rows.get();
        let width = key.shape.tokens.get();
        let mut data = StepData {
            tokens: vec![0; batch * width],
            pos: (0..batch).flat_map(|_| 0..width as i32).collect(),
            token_type: vec![0; batch * width],
            attention_mask: vec![0; batch * width],
            ..StepData::default()
        };
        let frame = entry
            .slots
            .iter()
            .find(|spec| spec.role == SlotRole::Pixels)
            .map(|spec| spec.shape[1..].iter().product::<usize>());
        for (i, row) in rows.iter().enumerate() {
            match *row {
                EncodeRow::Tokens { ids, types } => {
                    for (j, &id) in ids.iter().enumerate() {
                        data.tokens[i * width + j] = id as i32;
                        data.attention_mask[i * width + j] = 1;
                        if let Some(&kind) = types.get(j) {
                            data.token_type[i * width + j] = kind as i32;
                        }
                    }
                }
                EncodeRow::Patches { pixels, geometry } => {
                    if !entry.slots.iter().any(|spec| spec.role == SlotRole::Window) {
                        return Err(InvalidRequest::Encode(EncodeError::NoGeometry).into());
                    }
                    let want = frame.unwrap_or(0);
                    if pixels.len() != want {
                        return Err(InvalidRequest::Encode(EncodeError::FrameLength {
                            row: i,
                            got: pixels.len(),
                            want,
                        })
                        .into());
                    }
                    data.pixels.extend_from_slice(pixels);
                    data.pos_h.extend_from_slice(geometry.pos_h());
                    data.pos_w.extend_from_slice(geometry.pos_w());
                    data.window.extend_from_slice(geometry.window());
                    data.frame.extend_from_slice(geometry.frame());
                }
                EncodeRow::Frame { pixels, .. } => {
                    let want = frame.unwrap_or(0);
                    if pixels.len() != want {
                        return Err(InvalidRequest::Encode(EncodeError::FrameLength {
                            row: i,
                            got: pixels.len(),
                            want,
                        })
                        .into());
                    }
                    data.pixels.extend_from_slice(pixels);
                }
            }
        }

        let mut outputs = match self.execute(&entry, &data) {
            Ok(outputs) => outputs,
            Err(error) => return Err(self.recover_encode(entry.exe, error)),
        };
        let out = host_sample::read_logits(&mut outputs)?;
        let [out_rows, positions, out_width] = entry.output[..] else {
            unreachable!("an encode program returns [rows, positions, width]");
        };
        debug_assert_eq!(out_rows, batch);
        let per_row = positions * out_width;
        let rows = rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let kept = match *row {
                    // A hidden-state model returns every padded position; its real ones lead.
                    EncodeRow::Tokens { ids, .. } if positions == width => ids.len(),
                    _ => positions,
                };
                out[i * per_row..i * per_row + kept * out_width].to_vec()
            })
            .collect();
        self.refresh_caps_if_published();
        Ok(Encoded {
            width: out_width,
            rows,
        })
    }
}
