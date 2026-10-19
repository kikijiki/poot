//! The narrowing contract's range precondition, as an Intel NPU device enforces it (POOT-1178, POOT-1215).
//!
//! The policy is pure: it is generic over the buffer handle and reads contents through a closure, so it runs
//! (and its rows run) without a device or a driver. [`crate::VpuDevice`] is its one consumer.

use std::rc::Rc;

use poot_runtime_common::BufferRole;
use poot_target::{ElementKind, NarrowingContract};

/// Why the contract's range refused a value.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RangeViolation {
    /// An element of a constant's upload. `index` counts F32 words from the start of the buffer.
    Upload {
        index: usize,
        value: f32,
        max_magnitude: u32,
    },
    /// An upload into an F32 constant that does not cover whole words, so some word would go unchecked.
    Misaligned { offset: usize, len: usize },
    /// An element of a recorded execution's argument, read at a replay.
    Replay {
        key: String,
        role: &'static str,
        binding: usize,
        index: usize,
        value: f32,
        max_magnitude: u32,
    },
}

/// Whether a range-checked argument is read before the replay (an input) or after it (a result).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    Input,
    Result,
}

impl Direction {
    const fn name(self) -> &'static str {
        match self {
            Direction::Input => "input",
            Direction::Result => "result",
        }
    }
}

/// One F32 argument of a recorded command, checked against the narrowing contract's range. An input is read
/// before the replay and so is recorded only if no earlier command of the recording writes its buffer; a
/// result is read after the replay, which also covers a written buffer a later command reads.
pub(crate) struct RangeCheck<B> {
    key: String,
    binding: usize,
    direction: Direction,
    buffer: B,
    elements: usize,
}

/// The first element of `bytes`, read as little-endian F32 words, that `contract` does not admit, with its
/// index. A trailing partial word is not an element. This is the one reading of the range precondition: the
/// bytes of an upload and of a replay's inputs and results are checked through it.
fn first_outside(contract: &NarrowingContract, bytes: &[u8]) -> Option<(usize, f32)> {
    let (words, _) = bytes.as_chunks::<{ size_of::<f32>() }>();
    words
        .iter()
        .map(|&word| f32::from_le_bytes(word))
        .enumerate()
        .find(|&(_, value)| !contract.admits(value))
}

/// One argument of a dispatch the guard is shown: its buffer, the role the buffer was allocated for and
/// the element count the kernel binds.
pub(crate) struct GuardArg<B> {
    pub(crate) buffer: B,
    pub(crate) role: BufferRole,
    pub(crate) elements: usize,
}

/// The checks of the recording being built and, once per kind, what each costs:
///
/// - a constant (a weight, [`BufferRole::Weight`]) is checked from the bytes of its upload, once, and never
///   again: nothing writes it between replays except a recorded copy, whose destination is checked as that
///   copy's result;
/// - an input is read and checked before every replay, a result after it.
///
/// An upload into an F32 constant must cover whole words at a word-aligned offset, so no word goes unchecked.
pub(crate) struct RangeGuard<B> {
    contract: NarrowingContract,
    /// The checks of the open recording's commands.
    open: Vec<RangeCheck<B>>,
    /// The buffers a command already in the open recording writes (a graph execution's result, a copy's
    /// destination). Such a buffer holds stale data when a replay starts, so it is never range-checked as an
    /// input; it is checked as the result of its writer.
    written: Vec<B>,
    /// Constant uploads checked.
    pub(crate) constant_checks: u64,
    /// Input and result reads checked by replays.
    pub(crate) replay_checks: u64,
}

impl<B: Clone + PartialEq> RangeGuard<B> {
    pub(crate) fn new(contract: NarrowingContract) -> Self {
        Self {
            contract,
            open: Vec::new(),
            written: Vec::new(),
            constant_checks: 0,
            replay_checks: 0,
        }
    }

    /// Check the `bytes` just written at byte `offset` of a buffer of `role` and `element`: a weight of F32 is a
    /// constant, and its upload is the one time its range is read.
    pub(crate) fn uploaded(
        &mut self,
        role: BufferRole,
        element: ElementKind,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), RangeViolation> {
        if role != BufferRole::Weight || element != ElementKind::F32 {
            return Ok(());
        }
        let word = size_of::<f32>();
        if !offset.is_multiple_of(word) || !bytes.len().is_multiple_of(word) {
            return Err(RangeViolation::Misaligned {
                offset,
                len: bytes.len(),
            });
        }
        self.constant_checks += 1;
        match first_outside(&self.contract, bytes) {
            None => Ok(()),
            Some((index, value)) => Err(RangeViolation::Upload {
                index: offset / word + index,
                value,
                max_magnitude: self.contract.max_magnitude,
            }),
        }
    }

    /// Record the range checks of one graph execution: `inputs` then the one `output`, with the element kind
    /// the kernel binds each as.
    pub(crate) fn executed(
        &mut self,
        key: &str,
        inputs: &[(GuardArg<B>, ElementKind)],
        output: (GuardArg<B>, ElementKind),
    ) {
        let count = inputs.len();
        for (binding, (arg, element)) in inputs.iter().chain(std::iter::once(&output)).enumerate() {
            if *element != ElementKind::F32 {
                continue;
            }
            let direction = if binding < count {
                Direction::Input
            } else {
                Direction::Result
            };
            // A constant was checked when it was uploaded; an input an earlier command of this recording
            // writes holds nothing the host can check before the replay, its writer's result check covers it.
            if direction == Direction::Input
                && (arg.role == BufferRole::Weight || self.written.contains(&arg.buffer))
            {
                continue;
            }
            self.open.push(RangeCheck {
                key: key.to_owned(),
                binding,
                direction,
                buffer: arg.buffer.clone(),
                elements: arg.elements,
            });
        }
        self.written.push(output.0.buffer.clone());
    }

    /// Record a device-to-device copy into `dst`, `elements` words of `element`: its contents are only known
    /// when the recording runs, so an F32 destination is checked as the copy's result.
    pub(crate) fn copied(&mut self, dst: &B, element: ElementKind, elements: usize) {
        if element == ElementKind::F32 {
            self.open.push(RangeCheck {
                key: "copy".to_owned(),
                binding: 0,
                direction: Direction::Result,
                buffer: dst.clone(),
                elements,
            });
        }
        self.written.push(dst.clone());
    }

    /// Close the open recording's checks.
    pub(crate) fn finished(&mut self) -> Rc<[RangeCheck<B>]> {
        self.written.clear();
        std::mem::take(&mut self.open).into()
    }

    pub(crate) fn aborted(&mut self) {
        self.open.clear();
        self.written.clear();
    }

    /// Check every `direction` argument of `checks`, reading each buffer's contents through `read`.
    pub(crate) fn verify<E: From<RangeViolation>>(
        &mut self,
        checks: &[RangeCheck<B>],
        direction: Direction,
        mut read: impl FnMut(&B, usize) -> Result<Vec<u8>, E>,
    ) -> Result<(), E> {
        for check in checks.iter().filter(|c| c.direction == direction) {
            self.replay_checks += 1;
            let bytes = read(&check.buffer, check.elements * size_of::<f32>())?;
            if let Some((index, value)) = first_outside(&self.contract, &bytes) {
                return Err(RangeViolation::Replay {
                    key: check.key.clone(),
                    role: check.direction.name(),
                    binding: check.binding,
                    index,
                    value,
                    max_magnitude: self.contract.max_magnitude,
                }
                .into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FP16's range, as the NPU4 records it.
    const FP16: NarrowingContract = NarrowingContract {
        max_relative_error_ppm: 5_000,
        max_magnitude: 65_504,
    };

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A buffer handle the policy runs on without a device: an id, with the contents kept by the test.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct Id(u32);

    const WEIGHT: Id = Id(0);
    const INPUT: Id = Id(1);
    const OUTPUT: Id = Id(2);

    /// Reads through `verify` convert the violation; the rows read it as is.
    #[derive(Debug, PartialEq)]
    struct Read(RangeViolation);

    impl From<RangeViolation> for Read {
        fn from(violation: RangeViolation) -> Self {
            Read(violation)
        }
    }

    /// The range check reads little-endian F32 words and refuses the first one outside the range, at the edges
    /// of FP16: 65504 is admitted, 65504.5, its negative, NaN and both infinities are not; a trailing partial
    /// word is not an element.
    ///
    /// Mutation: report the last offender (`.rfind`) in `first_outside`; the index row goes red.
    #[test]
    fn the_first_word_outside_the_range_is_refused_at_the_edges_of_fp16() {
        for admitted in [0.0, 65_504.0, -65_504.0] {
            assert_eq!(first_outside(&FP16, &f32_bytes(&[admitted])), None);
        }
        for refused in [
            65_504.5,
            -65_504.5,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let (index, value) = first_outside(&FP16, &f32_bytes(&[1.0, refused, refused]))
                .unwrap_or_else(|| panic!("{refused} must be refused"));
            assert_eq!((index, value.to_bits()), (1, refused.to_bits()));
        }
        let mut bytes = f32_bytes(&[1.0]);
        bytes.extend_from_slice(&[0xff; 3]);
        assert_eq!(first_outside(&FP16, &bytes), None);
    }

    /// A guard with one recorded execution `(weight, input) -> output`, all F32.
    fn recorded() -> (RangeGuard<Id>, Rc<[RangeCheck<Id>]>) {
        let mut guard = RangeGuard::new(FP16);
        let arg = |buffer, role| GuardArg {
            buffer,
            role,
            elements: 4,
        };
        guard.executed(
            "k",
            &[
                (arg(WEIGHT, BufferRole::Weight), ElementKind::F32),
                (arg(INPUT, BufferRole::Input), ElementKind::F32),
            ],
            (arg(OUTPUT, BufferRole::Output), ElementKind::F32),
        );
        let ranges = guard.finished();
        (guard, ranges)
    }

    /// Constants are range-checked once per upload, inputs and results once per replay: three replays of a
    /// recording over one weight read the input and the result three times each and the weight not at all.
    ///
    /// Mutation: drop the `BufferRole::Weight` exemption in `RangeGuard::executed`; the weight joins the
    /// replay checks and the count goes from 6 to 9.
    #[test]
    fn constants_are_checked_once_per_upload_and_inputs_once_per_replay() {
        let (mut guard, ranges) = recorded();
        guard
            .uploaded(
                BufferRole::Weight,
                ElementKind::F32,
                0,
                &f32_bytes(&[1.0; 4]),
            )
            .unwrap();
        assert_eq!(guard.constant_checks, 1);
        for _ in 0..3 {
            for direction in [Direction::Input, Direction::Result] {
                guard
                    .verify(&ranges, direction, |_, bytes| Ok::<_, Read>(vec![0; bytes]))
                    .unwrap();
            }
        }
        assert_eq!(guard.replay_checks, 6, "input and result, three replays");
        assert_eq!(guard.constant_checks, 1, "no replay re-reads a constant");
    }

    /// A weight is refused at its upload with the element named from the start of the buffer, an F16 or
    /// non-weight upload is not read, and an input past the range is refused when a replay reads it.
    ///
    /// Mutation: make `uploaded` return `Ok(())` for a weight, or report the index chunk-relative (drop
    /// `offset / word`); the refusal row goes red.
    #[test]
    fn a_constant_past_the_range_is_refused_at_upload_and_an_input_at_replay() {
        let (mut guard, ranges) = recorded();
        let refused = guard
            .uploaded(
                BufferRole::Weight,
                ElementKind::F32,
                8,
                &f32_bytes(&[1.0, 65_504.5]),
            )
            .unwrap_err();
        assert_eq!(
            refused,
            RangeViolation::Upload {
                index: 3,
                value: 65_504.5,
                max_magnitude: 65_504
            },
            "word 1 of a chunk written at byte 8 is word 3 of the buffer"
        );
        guard
            .uploaded(
                BufferRole::Input,
                ElementKind::F32,
                0,
                &f32_bytes(&[f32::NAN]),
            )
            .expect("only a weight is a constant");
        guard
            .uploaded(BufferRole::Weight, ElementKind::F16, 0, &[0xff; 8])
            .expect("only F32 words are range-checked");

        let Read(refused) = guard
            .verify(&ranges, Direction::Input, |buffer, bytes| {
                assert_eq!(*buffer, INPUT, "the weight is not read");
                let mut contents = f32_bytes(&[0.0, 0.0, f32::INFINITY, 0.0]);
                contents.truncate(bytes);
                Ok::<_, Read>(contents)
            })
            .unwrap_err();
        assert!(
            matches!(
                refused,
                RangeViolation::Replay {
                    role: "input",
                    binding: 1,
                    index: 2,
                    ..
                }
            ),
            "{refused:?}"
        );
    }

    /// An F32 constant is written whole words at word-aligned offsets, or the upload is refused: a chunk that
    /// split a word would leave it unchecked.
    ///
    /// Mutation: skip the alignment test in `RangeGuard::uploaded`; the misaligned rows are admitted.
    #[test]
    fn an_upload_that_splits_a_word_of_a_constant_is_refused() {
        let (mut guard, _) = recorded();
        for (offset, len) in [(2, 8), (0, 6)] {
            assert_eq!(
                guard.uploaded(BufferRole::Weight, ElementKind::F32, offset, &vec![0; len]),
                Err(RangeViolation::Misaligned { offset, len })
            );
        }
    }

    /// An argument an earlier command of the recording writes is a result, not an input: it is not read before
    /// the replay, where it still holds the last replay's data.
    ///
    /// Mutation: skip the `written` test in `RangeGuard::executed`; the stale buffer is read as an input.
    #[test]
    fn a_buffer_written_inside_the_recording_is_never_read_as_an_input() {
        let mut guard = RangeGuard::new(FP16);
        let arg = |buffer| GuardArg {
            buffer,
            role: BufferRole::Activation,
            elements: 4,
        };
        let f32 = ElementKind::F32;
        guard.executed("first", &[(arg(INPUT), f32)], (arg(Id(7)), f32));
        guard.executed("second", &[(arg(Id(7)), f32)], (arg(OUTPUT), f32));
        let ranges = guard.finished();
        let mut read = Vec::new();
        guard
            .verify(&ranges, Direction::Input, |buffer, bytes| {
                read.push(*buffer);
                Ok::<_, Read>(vec![0; bytes])
            })
            .unwrap();
        assert_eq!(read, [INPUT]);
    }

    /// An aborted recording leaves no check behind for the next one.
    ///
    /// Mutation: leave `open` alone in `RangeGuard::aborted`; the next recording inherits the checks.
    #[test]
    fn an_aborted_recording_leaves_no_checks_behind() {
        let (mut guard, _) = recorded();
        let arg = |buffer| GuardArg {
            buffer,
            role: BufferRole::Input,
            elements: 4,
        };
        guard.executed(
            "k",
            &[(arg(INPUT), ElementKind::F32)],
            (arg(OUTPUT), ElementKind::F32),
        );
        guard.aborted();
        assert!(guard.finished().is_empty());
    }

    /// A weight filled by a recorded copy holds nothing the host can read at record time, so the copy's
    /// destination is range-checked as its result after the replay, and a graph that reads it takes no
    /// per-replay input check.
    ///
    /// Mutation: skip the check push in `RangeGuard::copied`; the out-of-range word is never found.
    #[test]
    fn a_constant_filled_by_a_recorded_copy_is_checked_as_the_copys_result() {
        let mut guard = RangeGuard::new(FP16);
        guard.copied(&WEIGHT, ElementKind::F32, 4);
        guard.executed(
            "k",
            &[(
                GuardArg {
                    buffer: WEIGHT,
                    role: BufferRole::Weight,
                    elements: 4,
                },
                ElementKind::F32,
            )],
            (
                GuardArg {
                    buffer: OUTPUT,
                    role: BufferRole::Output,
                    elements: 4,
                },
                ElementKind::F32,
            ),
        );
        let ranges = guard.finished();
        guard
            .verify(&ranges, Direction::Input, |_, _| -> Result<_, Read> {
                panic!("no input is read")
            })
            .unwrap();
        let Read(refused) = guard
            .verify(&ranges, Direction::Result, |buffer, _| {
                Ok::<_, Read>(if *buffer == WEIGHT {
                    f32_bytes(&[0.0, 0.0, 0.0, 70_000.0])
                } else {
                    f32_bytes(&[0.0; 4])
                })
            })
            .unwrap_err();
        assert!(
            matches!(&refused, RangeViolation::Replay { key, index: 3, .. } if key == "copy"),
            "{refused:?}"
        );
    }
}
