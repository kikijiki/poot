//! [`Driver::encode`]: one `encode` step over a batch of rows, returning each row's final hidden
//! states (or scores) for the host to pool or rank (POOT-739).
//!
//! A row is a token sequence (an embedding or reranking encoder, or a decoder's embedding pass), a set
//! of per-row request inputs (a vision tower's image values and grid), or both: [`EncodeRow`] is generic and
//! its values are checked against the model's schema like a served request's. A step runs on one entry
//! of its shape, prepared on first use like `generate`'s and kept under the same limits; token rows
//! are different: their entries are
//! admitted by [`Driver::prepare_encode`] before serving, at a fixed row count and every power-of-two
//! length up to the longest row, and a step runs on the smallest prepared length holding its longest
//! row (a padded position is masked, never attended to; a padded row is dropped from the answer). A
//! token request the prepared shapes cannot hold is a typed refusal, never a compile. An encode step
//! carries no cache any later step reads, so it refuses while a sequence is open (its cache would alias
//! the sequence's).

use std::num::NonZeroUsize;

use std::sync::Arc;

use poot_executor::{ExecError, ExecutableId};
use poot_models::io::{
    InputSet, InputValue, ModelSchema, Per, ProgramId, ProgramSpec, RequestInputs,
};
use poot_models::model::{LogitRows, StepShape, TraceError};

use super::bind::{self, RequestFrame, RowFeed, Side, StepColumns};
use super::caps::DriverCaps;
use super::error::{DriverError, EncodeError, InvalidOptions, InvalidRequest, Unsupported};
use super::host_sample;
use super::suffix::Head;
use super::{Driver, EntryKey, Feature, Incompatible, ModelRole, Program, State};

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

/// One row of an encode step: the token ids of an encoder's or a decoder's embedding pass, the values
/// of the inputs the `encode` program reads for it (a segment id per token, a tower's image values and
/// grid), or both. Its extent is its token count, else the leading extent of its per-row inputs, else
/// the positions the program fixes.
#[derive(Clone, Copy, Debug)]
pub struct EncodeRow<'a> {
    /// The row's token ids; `None` for a row of a program that reads none (a vision tower's).
    pub tokens: Option<&'a [u32]>,
    /// The values the row supplies for the program's request inputs.
    pub inputs: &'a RequestInputs,
}

/// What one [`Driver::encode`] step runs: every row of one extent.
#[derive(Clone, Copy, Debug)]
pub struct EncodeRequest<'a> {
    /// The rows of the step; every row has the same extent and carries the same inputs.
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

impl<S: State> Driver<S> {
    /// Turn a failed encode step into the driver's answer. The step graph's cache is scratch no later
    /// step reads, so a device failure that left the device usable (a fault, a failed transaction) is
    /// proven recoverable by resetting the executable's state: that is [`DriverError::EncodeStep`].
    /// A poisoned device, a failed reset or any other error is returned as it came.
    fn recover_encode(&mut self, exe: ExecutableId, error: DriverError) -> DriverError {
        let DriverError::Device(
            step @ (ExecError::Device(_)
            | ExecError::Fault { .. }
            | ExecError::Validation(_)
            | ExecError::NeedsReset(_)),
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
        if self
            .ledger
            .encode_shapes
            .is_some_and(|held| held != *shapes)
        {
            return Err(InvalidOptions::EncodeShapesChanged.into());
        }
        for len in shapes.lengths() {
            self.prepared(Self::token_key(shapes.rows, len))?;
        }
        self.ledger.encode_shapes = Some(*shapes);
        Ok(self.refresh_caps())
    }

    /// [`Driver::prepare_encode`] for a model that may not encode at all (a generative family whose body
    /// has no embedding pass, or a placement with no form of an encode step): `Ok(false)` when the
    /// family or the placement refuses it, with nothing prepared; any other refusal is the error.
    pub fn prepare_encode_if_traced(&mut self, shapes: &EncodeShapes) -> Result<bool, DriverError> {
        match self.prepare_encode(shapes) {
            Ok(_) => Ok(true),
            Err(DriverError::Unsupported(
                Unsupported::Trace(TraceError::ProgramUnsupported { .. })
                | Unsupported::Incompatible(Incompatible {
                    first: Feature::Encode,
                    ..
                }),
            )) => Ok(false),
            Err(other) => Err(other),
        }
    }

    /// The entry key of an encode step of `rows` rows of `tokens` positions each, carrying the
    /// carriable inputs `carried` and, when the rows hold padded per-row inputs, their leading extent.
    fn encode_key(
        rows: NonZeroUsize,
        tokens: NonZeroUsize,
        row_extent: Option<NonZeroUsize>,
        carried: InputSet,
    ) -> EntryKey {
        EntryKey {
            role: ModelRole::Target,
            program: Program::Encode,
            shape: StepShape::new(rows, tokens, tokens)
                .with_logits(LogitRows::All)
                .with_row_extent(row_extent),
            head: Head::Logits,
            adapters: None,
            carried,
        }
    }

    fn token_key(rows: NonZeroUsize, tokens: NonZeroUsize) -> EntryKey {
        Self::encode_key(rows, tokens, None, InputSet::EMPTY)
    }

    /// Run one encode step over `request`'s rows.
    pub fn encode(&mut self, request: &EncodeRequest<'_>) -> Result<Encoded, DriverError> {
        let EncodeRequest { rows } = *request;
        if rows.is_empty() {
            return Err(InvalidRequest::Encode(EncodeError::Empty).into());
        }
        if self.state.open_sequences() > 0 {
            return Err(InvalidRequest::Encode(EncodeError::SequenceOpen).into());
        }
        let n = rows.len();
        let handle = Arc::clone(&self.handle);
        let schema = handle.model().schema();
        let program =
            schema
                .encode()
                .ok_or(Unsupported::Trace(TraceError::ProgramUnsupported {
                    family: handle.config().family,
                    program: ProgramId::ENCODE,
                }))?;
        // Each row's extent: its tokens, its padded per-row inputs' leading axis, else the positions
        // the program fixes. Checked once, here, like a served request's inputs.
        let mut extents = Vec::with_capacity(n);
        for (row, entry) in rows.iter().enumerate() {
            let tokens = entry.tokens.map_or(0, <[u32]>::len);
            if entry.tokens.is_some() && tokens == 0 {
                return Err(InvalidRequest::Encode(EncodeError::EmptyRow { row }).into());
            }
            schema
                .check_request(program, entry.inputs, tokens)
                .map_err(InvalidRequest::Input)?;
            let padded = padded_extent(schema, program, entry.inputs)
                .map_err(|()| InvalidRequest::Encode(EncodeError::RaggedRow { row }))?;
            let positions = if tokens > 0 {
                tokens
            } else {
                padded
                    .or(program.positions().map(NonZeroUsize::get))
                    .ok_or(InvalidRequest::Encode(EncodeError::EmptyRow { row }))?
            };
            // A row of no positions has nothing to encode (and no entry shape).
            if positions == 0 {
                return Err(InvalidRequest::Encode(EncodeError::EmptyRow { row }).into());
            }
            extents.push((positions, padded));
        }
        let with_tokens = rows[0].tokens.is_some();
        if rows.iter().any(|row| row.tokens.is_some() != with_tokens) {
            return Err(InvalidRequest::Encode(EncodeError::Mixed).into());
        }
        let row_extent = extents[0].1.and_then(NonZeroUsize::new);
        if extents
            .iter()
            .any(|&(_, padded)| padded.and_then(NonZeroUsize::new) != row_extent)
        {
            return Err(InvalidRequest::Encode(EncodeError::Mixed).into());
        }
        let fed: Vec<usize> = extents.iter().map(|&(positions, _)| positions).collect();
        let feeds: Vec<RowFeed<'_>> = rows
            .iter()
            .zip(&fed)
            .map(|(row, &fed)| RowFeed {
                inputs: row.inputs,
                start: 0,
                fed,
            })
            .collect();
        let carried = bind::carried(schema, program, &feeds);
        // A token step runs at the prepared rows and the prepared length that holds its longest row (the
        // rest pad); any other step at the request's own rows and extent.
        let (key, entry) = if with_tokens {
            let shapes = self
                .ledger
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
            for (row, &tokens) in fed.iter().enumerate() {
                if tokens > shapes.max_len.get() {
                    return Err(InvalidRequest::Encode(EncodeError::RowTooLong {
                        row,
                        tokens,
                        max: shapes.max_len.get(),
                    })
                    .into());
                }
                longest = longest.max(tokens);
            }
            let len = shapes
                .lengths()
                .into_iter()
                .find(|len| len.get() >= longest)
                .expect("the longest length is max_len, which holds every checked row");
            let key = Self::encode_key(shapes.rows, len, row_extent, carried);
            let entry = self
                .ledger
                .entries
                .get(&key)
                .cloned()
                .ok_or(InvalidRequest::Encode(EncodeError::NotPrepared))?;
            (key, entry)
        } else {
            let positions = NonZeroUsize::new(fed[0]).expect("a row has a position");
            if fed.iter().any(|&f| f != positions.get()) {
                return Err(InvalidRequest::Encode(EncodeError::Mixed).into());
            }
            let rows = NonZeroUsize::new(n).expect("checked non-empty");
            let key = Self::encode_key(rows, positions, row_extent, carried);
            let entry = self.prepared(key)?;
            (key, entry)
        };
        let batch = key.shape.rows.get();
        let width = key.shape.tokens.get();
        let mut columns = StepColumns {
            tokens: vec![0; batch * width],
            pos: (0..batch).flat_map(|_| 0..width as i32).collect(),
            real: vec![0; batch * width],
            ..StepColumns::default()
        };
        for (i, row) in rows.iter().enumerate() {
            let held = fed[i];
            for j in 0..held {
                columns.real[i * width + j] = 1;
                if let Some(tokens) = row.tokens {
                    columns.tokens[i * width + j] = tokens[j] as i32;
                }
            }
        }
        let frame = RequestFrame {
            rows: batch,
            tokens: width,
            side: Side::Right,
            feeds: &feeds,
        };
        let mut outputs = match self.execute(&entry, &columns, &frame) {
            Ok(outputs) => outputs,
            Err(error) => return Err(self.recover_encode(entry.exe, error)),
        };
        let out = host_sample::read_logits(&mut outputs)?;
        let [out_rows, positions, out_width] = entry.output[..] else {
            unreachable!("an encode program returns [rows, positions, width]");
        };
        debug_assert_eq!(out_rows, batch);
        let per_row = positions * out_width;
        let rows = fed
            .iter()
            .enumerate()
            .map(|(i, &held)| {
                // A hidden-state model returns every padded position; its real ones lead.
                let kept = if with_tokens && positions == width {
                    held
                } else {
                    positions
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

/// The leading extent of the padded per-row inputs `inputs` holds for `program`: `None` when it holds
/// none, `Err` when two of them disagree.
fn padded_extent(
    schema: &ModelSchema,
    program: &ProgramSpec,
    inputs: &RequestInputs,
) -> Result<Option<usize>, ()> {
    let mut extent = None;
    for (id, _) in program.reads() {
        let Some(input) = schema.input(*id) else {
            continue;
        };
        let Per::Row { padded: true, .. } = input.form().per() else {
            continue;
        };
        let Some(InputValue::Row(tensor)) = inputs.get(*id) else {
            continue;
        };
        let leading = tensor.shape().first().copied().unwrap_or(0);
        if extent.is_some_and(|held| held != leading) {
            return Err(());
        }
        extent = Some(leading);
    }
    Ok(extent)
}
