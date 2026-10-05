//! Backend-neutral helpers shared by poot GPU runtimes.

mod kernel;
mod memory;
pub mod telemetry;
pub use kernel::{
    ArgAccess, ArgSchema, CompiledKernel, KernelArgError, KernelCode, check_element_schema,
    check_kernel_args,
};
pub use memory::{AllocGuard, BufferRole, MemoryCounterSnapshot, MemoryCounters};
pub use telemetry::{
    CallCounts, CallPurpose, Coverage, ExecutionCounters, TransferCounts, TransferDirection,
};

/// Failure while converting a logical element count into a native buffer layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutError {
    /// The logical element count does not fit the native u32 length ABI.
    ElementCountTooLarge,
    /// Multiplying the element count by its storage width overflowed `usize`.
    ByteSizeOverflow,
}

/// Validate the native u32 element-count ABI and compute the allocation size in bytes.
pub fn checked_layout(elements: usize, element_bytes: usize) -> Result<(u32, usize), LayoutError> {
    let elem_count = u32::try_from(elements).map_err(|_| LayoutError::ElementCountTooLarge)?;
    let byte_capacity = elements
        .checked_mul(element_bytes)
        .ok_or(LayoutError::ByteSizeOverflow)?;
    Ok((elem_count, byte_capacity))
}

/// A validated byte range inside a native buffer: the span a backend copy may touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyRange {
    /// Byte offset from the start of the allocation.
    pub byte_offset: usize,
    /// Number of bytes in the range.
    pub byte_len: usize,
}

/// Failure while validating a copy range against a native buffer's logical length and capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyRangeError {
    /// `whole_buffer` was requested but the elements do not span exactly the whole buffer.
    SizeMismatch {
        /// Logical element count of the buffer.
        have: usize,
        /// Element count the caller asked to copy.
        got: usize,
    },
    /// Converting the element range to bytes overflowed `usize`.
    RangeOverflow,
    /// The computed byte end exceeds the buffer's byte capacity.
    RangeOutOfBounds {
        /// Byte offset where the range starts.
        byte_offset: usize,
        /// Byte offset one past the range's last byte.
        byte_end: usize,
        /// Buffer byte capacity the range was checked against.
        byte_capacity: usize,
    },
}

/// Validate an element range against a buffer and compute the byte range a copy may touch.
///
/// When `whole_buffer` is set, the request must start at element offset 0 and cover exactly
/// `elem_count` elements. The resulting span `[byte_offset, byte_offset + byte_len)` must end at
/// or before `byte_capacity` (a zero-length range at the capacity boundary is valid).
pub fn checked_copy_range(
    element_offset: usize,
    elements: usize,
    element_bytes: usize,
    whole_buffer: bool,
    elem_count: u32,
    byte_capacity: usize,
) -> Result<CopyRange, CopyRangeError> {
    if whole_buffer && (element_offset != 0 || elements != elem_count as usize) {
        return Err(CopyRangeError::SizeMismatch {
            have: elem_count as usize,
            got: elements,
        });
    }
    let byte_offset = element_offset
        .checked_mul(element_bytes)
        .ok_or(CopyRangeError::RangeOverflow)?;
    let byte_len = elements
        .checked_mul(element_bytes)
        .ok_or(CopyRangeError::RangeOverflow)?;
    let byte_end = byte_offset
        .checked_add(byte_len)
        .ok_or(CopyRangeError::RangeOverflow)?;
    if byte_end > byte_capacity {
        return Err(CopyRangeError::RangeOutOfBounds {
            byte_offset,
            byte_end,
            byte_capacity,
        });
    }
    Ok(CopyRange {
        byte_offset,
        byte_len,
    })
}

/// A device backend whose availability a test run can make mandatory.
///
/// Device tests skip when their backend cannot open. A skip reads as a pass, so a broken environment
/// (no driver, wrong shell, no GPU) reports green while running nothing. `POOT_REQUIRE_<BACKEND>=1`
/// turns that skip into a panic. This type is the one place that reads the switch: context opens
/// ([`Self::fail_if_required`]) decide through [`Self::required`]; device tests build their own skip
/// path on top of [`Self::fail_if_required`] (see `poot_test_util::device_skip`, card 671 - this
/// crate's own `skip_unavailable`/`open_or_skip` had no production caller, only that device-test
/// pattern).
/// `scripts/with-required-backend.sh` sets exactly one backend's variable per lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceBackend {
    /// The wgpu executor (`poot-gpu`, `poot-runtime`).
    Wgpu,
    /// The Vulkan runtime (`poot-vulkan-runtime`).
    Vulkan,
    /// The ROCm/HSA runtime (`poot-rocm-runtime`, `poot-rocm-gpu`).
    Rocm,
    /// The NVIDIA PTX runtime (`poot-ptx-runtime`, `poot-ptx-gpu`).
    Ptx,
}

impl DeviceBackend {
    /// Every backend, for tests and scripts that must cover all of them.
    pub const ALL: [Self; 4] = [Self::Wgpu, Self::Vulkan, Self::Rocm, Self::Ptx];

    /// The environment variable that requires this backend (`POOT_REQUIRE_<BACKEND>`).
    pub const fn variable(self) -> &'static str {
        match self {
            Self::Wgpu => "POOT_REQUIRE_WGPU",
            Self::Vulkan => "POOT_REQUIRE_VULKAN",
            Self::Rocm => "POOT_REQUIRE_ROCM",
            Self::Ptx => "POOT_REQUIRE_PTX",
        }
    }

    /// The name used in skip and failure messages.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Wgpu => "wgpu",
            Self::Vulkan => "Vulkan",
            Self::Rocm => "ROCm",
            Self::Ptx => "PTX",
        }
    }

    /// Whether this backend is required: its own variable is `1`.
    pub fn required(self, get: impl Fn(&str) -> Option<String>) -> bool {
        get(self.variable()).is_some_and(|value| value == "1")
    }

    /// [`Self::required`] read from the process environment.
    #[cfg(test)]
    pub(crate) fn required_in_env(self) -> bool {
        self.required(|name| std::env::var(name).ok())
    }

    /// Panic when this backend is required. The one place the failure message is written.
    fn panic_if_required(
        self,
        get: impl Fn(&str) -> Option<String>,
        reason: &dyn std::fmt::Display,
    ) {
        if self.required(get) {
            panic!(
                "required {} device unavailable instead of skipping ({}=1): {reason}",
                self.label(),
                self.variable(),
            );
        }
    }

    /// Pass a failed context open through, or panic when the backend is required. Context-open code
    /// wraps its error in this so a skip guard further up never sees an `Err` it would turn into a
    /// silent pass. `get` reads the environment (injected so a test can set only one variable).
    pub fn fail_if_required<E: std::fmt::Display>(
        self,
        get: impl Fn(&str) -> Option<String>,
        error: E,
    ) -> E {
        self.panic_if_required(get, &error);
        error
    }
}

/// Fold a 1-D workgroup grid so the X dimension stays within `limit_x` (and the resulting Y within
/// `limit_y`). A pure 1-D grid `[n, 1, 1]` with `n > limit_x` becomes
/// `[limit_x, ceil(n / limit_x), 1]`. A grid that already has `y > 1` or `z > 1` is refused when
/// `x` is over the cap (`None`): the caller must keep the plan-level 2-D layout that already
/// respects the cap. Returns the input unchanged when `x` already fits. Returns `None` when the
/// fold cannot stay within both limits (or the input is a non-1-D grid over the X cap), so the
/// caller reports GridCap instead of dispatching an illegal grid.
///
/// This is the single host-side fold for wgpu/Vulkan dispatch sizing. The kernel side reconstructs
/// the linear index as `group_y * (folded_x * workgroup_size) + global_id.x` in shared codegen
/// (`poot-codegen`'s `thread_index` emission for SpirvVulkan), with the usual tail bounds guard.
pub fn fold_grid(groups: [u32; 3], limit_x: u32, limit_y: u32) -> Option<[u32; 3]> {
    let (x, y, z) = (groups[0], groups[1], groups[2]);
    if x <= limit_x {
        return Some(groups);
    }
    if y != 1 || z != 1 {
        // Already multi-dimensional and still over the X cap: no safe further fold without a Z scheme.
        return None;
    }
    let x_fold = limit_x.max(1);
    let y_fold = x.div_ceil(x_fold);
    if y_fold > limit_y.max(1) {
        return None;
    }
    Some([x_fold, y_fold, 1])
}

/// FNV-1a (64-bit) over bytes: the one deterministic fingerprint behind every kernel-cache key, HSACO/PTX
/// entry digest and backend cache key. Every crate that needs a content fingerprint imports this one
/// rather than keeping its own copy of the loop (cards 211/226, 501/608).
pub fn fnv1a_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// FNV-1a (64-bit) over a string: a deterministic content fingerprint for backend cache keys.
pub fn fnv1a(s: &str) -> u64 {
    fnv1a_bytes(s.as_bytes())
}

/// Round an f32 to bf16 bits (round-to-nearest, ties-to-even) and return the 16-bit pattern.
/// NaN keeps its sign: a quiet NaN stays quiet with its high payload bits, and a signaling NaN is
/// quieted (bf16 has no signaling NaN). Infinities, the sign, and the exponent range carry through;
/// values above bf16's max finite round to infinity.
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        // Keep it a NaN: force the quiet bit (bf16's top mantissa bit) and preserve sign + payload.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_bytes_matches_the_published_test_vectors() {
        assert_eq!(fnv1a_bytes(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a_bytes(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a_bytes(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn checked_layout_reports_each_overflow() {
        if let Some(elements) = (u32::MAX as usize).checked_add(1) {
            assert_eq!(
                checked_layout(elements, 1),
                Err(LayoutError::ElementCountTooLarge)
            );
        }
        assert_eq!(
            checked_layout(2, usize::MAX),
            Err(LayoutError::ByteSizeOverflow)
        );
    }

    /// A whole-buffer request must start at 0 and cover exactly `elem_count`. Both a wrong count
    /// and a nonzero offset are size mismatches; the check runs before any byte arithmetic, so an
    /// overflowing request against a wrong count reports the mismatch.
    #[test]
    fn checked_copy_range_reports_whole_buffer_size_mismatch() {
        assert_eq!(
            checked_copy_range(0, 3, 4, true, 2, 8),
            Err(CopyRangeError::SizeMismatch { have: 2, got: 3 })
        );
        assert_eq!(
            checked_copy_range(1, 2, 4, true, 2, 8),
            Err(CopyRangeError::SizeMismatch { have: 2, got: 2 })
        );
        assert_eq!(
            checked_copy_range(usize::MAX, usize::MAX, 4, true, 4, 16),
            Err(CopyRangeError::SizeMismatch {
                have: 4,
                got: usize::MAX
            })
        );
        assert_eq!(
            checked_copy_range(0, 4, 4, true, 4, 16),
            Ok(CopyRange {
                byte_offset: 0,
                byte_len: 16
            })
        );
    }

    /// Each stage of the byte-range arithmetic reports [`CopyRangeError::RangeOverflow`]:
    /// offset*width, length*width, and offset+len (the third needs width 1 so both muls succeed).
    #[test]
    fn checked_copy_range_reports_range_overflow() {
        assert_eq!(
            checked_copy_range(usize::MAX, 1, 4, false, 4, 16),
            Err(CopyRangeError::RangeOverflow)
        );
        assert_eq!(
            checked_copy_range(0, usize::MAX, 2, false, 4, 8),
            Err(CopyRangeError::RangeOverflow)
        );
        assert_eq!(
            checked_copy_range(usize::MAX, 1, 1, false, 4, 8),
            Err(CopyRangeError::RangeOverflow)
        );
    }

    /// `[4, 12)` against an 8-byte capacity: the error carries the computed span literally.
    #[test]
    fn checked_copy_range_reports_out_of_bounds() {
        assert_eq!(
            checked_copy_range(1, 2, 4, false, 4, 8),
            Err(CopyRangeError::RangeOutOfBounds {
                byte_offset: 4,
                byte_end: 12,
                byte_capacity: 8
            })
        );
    }

    /// Boundary: end == capacity is accepted, end == capacity + 1 is rejected.
    #[test]
    fn checked_copy_range_accepts_capacity_and_rejects_capacity_plus_one() {
        assert_eq!(
            checked_copy_range(0, 8, 1, false, 8, 8),
            Ok(CopyRange {
                byte_offset: 0,
                byte_len: 8
            })
        );
        assert_eq!(
            checked_copy_range(0, 8, 1, false, 8, 7),
            Err(CopyRangeError::RangeOutOfBounds {
                byte_offset: 0,
                byte_end: 8,
                byte_capacity: 7
            })
        );
    }

    /// Zero-length edges: an empty range ending exactly at capacity is valid, an empty range
    /// starting past capacity is out of bounds, and a whole-buffer zero request against a
    /// non-empty buffer is a size mismatch.
    #[test]
    fn checked_copy_range_zero_length_edges() {
        assert_eq!(
            checked_copy_range(2, 0, 4, false, 4, 8),
            Ok(CopyRange {
                byte_offset: 8,
                byte_len: 0
            })
        );
        assert_eq!(
            checked_copy_range(3, 0, 4, false, 4, 8),
            Err(CopyRangeError::RangeOutOfBounds {
                byte_offset: 12,
                byte_end: 12,
                byte_capacity: 8
            })
        );
        assert_eq!(
            checked_copy_range(0, 0, 4, true, 4, 16),
            Err(CopyRangeError::SizeMismatch { have: 4, got: 0 })
        );
    }

    /// Literal expected dims for [`fold_grid`]: below / at / over the X cap, `limit*k + r`, and a
    /// huge 1-D count. Every expectation is written out, not derived from the function under test.
    #[test]
    fn fold_grid_literal_table() {
        const L: u32 = 65_535;
        // Below the cap: unchanged.
        assert_eq!(fold_grid([1, 1, 1], L, L), Some([1, 1, 1]));
        assert_eq!(fold_grid([65_534, 1, 1], L, L), Some([65_534, 1, 1]));
        // At the cap: unchanged (x is legal).
        assert_eq!(fold_grid([65_535, 1, 1], L, L), Some([65_535, 1, 1]));
        // One over: fold to x=limit, y=2.
        assert_eq!(fold_grid([65_536, 1, 1], L, L), Some([65_535, 2, 1]));
        // limit*1 + r (r in 1..limit): y=2.
        assert_eq!(fold_grid([65_535 + 100, 1, 1], L, L), Some([65_535, 2, 1]));
        // Exact limit*2: y=2 (div_ceil of 2*limit by limit).
        assert_eq!(fold_grid([131_070, 1, 1], L, L), Some([65_535, 2, 1]));
        // limit*2 + 1: y=3.
        assert_eq!(fold_grid([131_071, 1, 1], L, L), Some([65_535, 3, 1]));
        // The olmo2 isl=2048 repro grid (65536 workgroups).
        assert_eq!(fold_grid([65_536, 1, 1], L, L), Some([65_535, 2, 1]));
        // Huge: y = ceil(n / limit).
        assert_eq!(fold_grid([1_000_000, 1, 1], L, L), Some([65_535, 16, 1]));
        // Already 2-D under the cap: unchanged (plan-level fold stays).
        assert_eq!(fold_grid([65_535, 3, 1], L, L), Some([65_535, 3, 1]));
        // Already 2-D over the X cap: refuse (no safe further fold).
        assert_eq!(fold_grid([65_536, 2, 1], L, L), None);
        // Folded Y would exceed limit_y: refuse.
        assert_eq!(fold_grid([u32::MAX, 1, 1], L, 1), None);
        // Non-1-D with a legal X: unchanged even when Z > 1.
        assert_eq!(fold_grid([10, 1, 4], L, L), Some([10, 1, 4]));
    }

    /// Golden table for [`f32_to_bf16`]. Every expectation is a hex literal, never computed by the
    /// function under test. Covers exact ties (round to even both directions), values just above and
    /// below the midpoint, max-finite overflow to infinity, +/-inf, qNaN and sNaN staying NaN,
    /// +/-0, and subnormals (including a tie at the subnormal/normal boundary).
    #[test]
    fn f32_to_bf16_golden_table() {
        const CASES: &[(u32, u16)] = &[
            // Exact ties: low16 == 0x8000, round to even (even result both down and up).
            (0x3F80_8000, 0x3F80),
            (0x3F81_8000, 0x3F82),
            (0xBF80_8000, 0xBF80),
            (0xBF81_8000, 0xBF82),
            // Just below / just above the same midpoints.
            (0x3F80_7FFF, 0x3F80),
            (0x3F80_8001, 0x3F81),
            (0x3F81_7FFF, 0x3F81),
            (0x3F81_8001, 0x3F82),
            (0xBF80_7FFF, 0xBF80),
            (0xBF80_8001, 0xBF81),
            // Above the midpoint but far from it (the value the old ROCm body got wrong).
            (0x3F80_A000, 0x3F81),
            (0x3F80_9FFF, 0x3F81),
            // Exactly representable: identity on the high 16 bits.
            (0x3F80_0000, 0x3F80),
            (0xC000_0000, 0xC000),
            (0x3F81_0000, 0x3F81),
            // Max finite overflows to infinity; a tie at the top boundary goes to the even side.
            (0x7F7F_FFFF, 0x7F80),
            (0xFF7F_FFFF, 0xFF80),
            (0x7F7F_8000, 0x7F80),
            (0x7F7E_8000, 0x7F7E),
            // +/-infinity.
            (0x7F80_0000, 0x7F80),
            (0xFF80_0000, 0xFF80),
            // Quiet NaN stays quiet NaN, sign and high payload bits preserved.
            (0x7FC0_0000, 0x7FC0),
            (0x7FD2_ABCD, 0x7FD2),
            (0xFFC0_0000, 0xFFC0),
            // Signaling NaN is quieted (bf16 has no sNaN), sign preserved.
            (0x7F80_0001, 0x7FC0),
            (0xFF80_0001, 0xFFC0),
            (0x7FBF_FFFF, 0x7FFF),
            // +/-0.
            (0x0000_0000, 0x0000),
            (0x8000_0000, 0x8000),
            // Subnormals: flush to zero, tie to even, just off the tie, and the sub/normal boundary.
            (0x0000_0001, 0x0000),
            (0x8000_0001, 0x8000),
            (0x0000_8000, 0x0000),
            (0x0000_8001, 0x0001),
            (0x0001_8000, 0x0002),
            (0x007F_FFFF, 0x0080),
            (0x007F_8000, 0x0080),
        ];
        for &(bits, want) in CASES {
            let got = f32_to_bf16(f32::from_bits(bits));
            assert_eq!(got, want, "f32_to_bf16({bits:#010x}) = {got:#06x}");
        }
    }
}

#[cfg(test)]
mod device_backend_tests {
    use super::DeviceBackend;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    /// An environment that sets exactly the named variables to `1`.
    fn env_with(names: &'static [&'static str]) -> impl Fn(&str) -> Option<String> {
        move |name| names.contains(&name).then(|| "1".to_string())
    }

    fn panic_message(run: impl FnOnce()) -> Option<String> {
        let payload = catch_unwind(AssertUnwindSafe(run)).err()?;
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()));
        Some(message.expect("device backend panics carry a message"))
    }

    /// A backend is required by its own variable and by nothing else: not another backend's variable and
    /// not the retired all-backend `POOT_REQUIRE_GPU`. This is the isolation
    /// `scripts/with-required-backend.sh` relies on.
    #[test]
    fn each_backend_is_required_by_its_own_variable_only() {
        let variables = [
            "POOT_REQUIRE_WGPU",
            "POOT_REQUIRE_VULKAN",
            "POOT_REQUIRE_ROCM",
            "POOT_REQUIRE_PTX",
        ];
        for (backend, own) in DeviceBackend::ALL.into_iter().zip(variables) {
            assert_eq!(backend.variable(), own);
            assert!(!backend.required(|_| None), "{backend:?} unset");
            assert!(
                backend.required(|name| (name == own).then(|| "1".to_string())),
                "{backend:?} with only {own}=1"
            );
            assert!(
                !backend.required(|name| (name == "POOT_REQUIRE_GPU").then(|| "1".to_string())),
                "{backend:?} must ignore the retired POOT_REQUIRE_GPU=1"
            );
            for other in variables.into_iter().filter(|other| *other != own) {
                assert!(
                    !backend.required(|name| (name == other).then(|| "1".to_string())),
                    "{backend:?} must not be required by {other}=1"
                );
            }
            for value in ["0", "true", "yes", ""] {
                assert!(
                    !backend.required(|name| (name == own).then(|| value.to_string())),
                    "{backend:?} must only accept the value 1, not {value:?}"
                );
            }
        }
    }

    // card 671: the three tests that used to live here (a device that opened is returned even when
    // required, a missing device skips when not required, skip_unavailable panics only when
    // required) exercised skip_unavailable_with/open_or_skip_with directly - both deleted (dead
    // production code, moved to poot_test_util::device_skip, whose own tests now cover this
    // Option-wrapping/skip-printing behavior). The panic-when-required half of that behavior is
    // fail_if_required's, covered below by fail_if_required_passes_the_error_through_unless_required.

    /// The model-free lane (`scripts/test-model-free.sh`) names each `POOT_REQUIRE_<BACKEND>` by hand. A
    /// device test classed model-free must fail there, not skip, so a backend added to
    /// [`DeviceBackend::ALL`] and left out of that list would skip silently. The lane sets
    /// `POOT_MODEL_FREE_LANE=1`; under it every backend must read as required. Outside the lane this
    /// has nothing to check.
    #[test]
    fn the_model_free_lane_requires_every_backend() {
        if std::env::var("POOT_MODEL_FREE_LANE").as_deref() != Ok("1") {
            return;
        }
        for backend in DeviceBackend::ALL {
            assert!(
                backend.required_in_env(),
                "the model-free lane does not set {}=1: add it to lane_environment in scripts/test-model-free.sh",
                backend.variable()
            );
        }
    }

    /// The context-open guard hands the error back untouched unless the backend is required.
    #[test]
    fn fail_if_required_passes_the_error_through_unless_required() {
        assert_eq!(
            DeviceBackend::Rocm.fail_if_required(|_| None, "no agent"),
            "no agent"
        );
        let message = panic_message(|| {
            DeviceBackend::Rocm.fail_if_required(env_with(&["POOT_REQUIRE_ROCM"]), "no agent");
        })
        .expect("a required backend must fail the open");
        assert!(
            message.contains("required ROCm device unavailable"),
            "{message}"
        );
        assert!(message.contains("POOT_REQUIRE_ROCM=1"), "{message}");
        assert!(message.contains("no agent"), "{message}");
    }
}
