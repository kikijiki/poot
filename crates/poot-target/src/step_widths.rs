//! The step widths an NPU's region artifacts are compiled for (POOT-1278, design npu-prefill-handoff-1267
//! section 5.9).
//!
//! An NPU artifact has static shapes, so one compile serves one step width. The widths a device prepares are
//! a small power-of-two set (OpenVINO's NPUW chunks a prompt the same way), configured by the operator that
//! opens the device and carried by the target's capabilities, so the planner and the driver read them from
//! the target and no constant names a chunk size.

use std::fmt;

/// A set of step widths, each a power of two: a bitset in which bit `i` is the width `2^i`. `Copy`, so it
/// travels inside the capability structs.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StepWidths(u64);

/// A width that cannot be in a [`StepWidths`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, thiserror::Error)]
#[error("step width {width} is not a power of two")]
pub struct NotAPowerOfTwo {
    /// The refused width (zero included).
    pub width: usize,
}

impl StepWidths {
    /// No width: a device with none configured prepares no step.
    pub const NONE: StepWidths = StepWidths(0);

    /// The set of `widths`.
    ///
    /// # Errors
    ///
    /// [`NotAPowerOfTwo`] naming the first width that is zero or not a power of two.
    pub fn new(widths: impl IntoIterator<Item = usize>) -> Result<Self, NotAPowerOfTwo> {
        let mut bits = 0u64;
        for width in widths {
            if !width.is_power_of_two() {
                return Err(NotAPowerOfTwo { width });
            }
            bits |= 1u64 << width.trailing_zeros();
        }
        Ok(StepWidths(bits))
    }

    /// Whether `width` is in the set.
    pub fn contains(self, width: usize) -> bool {
        width.is_power_of_two() && self.0 & (1u64 << width.trailing_zeros()) != 0
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The widths, smallest first.
    pub fn iter(self) -> impl Iterator<Item = usize> {
        (0..u64::BITS)
            .filter(move |bit| self.0 & (1u64 << bit) != 0)
            .map(|bit| 1usize << bit)
    }
}

impl fmt::Debug for StepWidths {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The set holds exactly the powers of two it was built from, lists them smallest first, and refuses a
    /// width that is not one.
    /// Mutation: build the set with `width.next_power_of_two()` in place of the check; the refusal rows go red.
    #[test]
    fn a_set_holds_exactly_its_power_of_two_widths() {
        let set = StepWidths::new([128, 32, 128]).unwrap();
        assert_eq!(set.iter().collect::<Vec<_>>(), [32, 128]);
        assert!(set.contains(32) && set.contains(128));
        for absent in [0, 1, 31, 33, 64, 127, 129, 256, usize::MAX] {
            assert!(!set.contains(absent), "{absent}");
        }
        assert!(StepWidths::NONE.is_empty() && !StepWidths::NONE.contains(1));
        for refused in [0, 3, 96, 129] {
            assert_eq!(
                StepWidths::new([64, refused]),
                Err(NotAPowerOfTwo { width: refused })
            );
        }
    }
}
