//! [`MatrixUnits`]: the matrix-multiply fragments a GPU exposes, one [`FragmentConfig`] per operand dtype
//! (POOT-1431, `receipts/1429-narrow-operands/design.md` section 4.1). Runtimes report them from the
//! device; the planner's operand-precision pass and the fragment micro-tile read only this, never a gfx
//! code, an SM number or a device name.
//!
//! A fragment multiplies two operands of one narrow dtype and accumulates in f32. Only the 16x16x16 shape
//! lowers today, so a device offers at most one config per dtype and [`MatrixUnits`] stores them by dtype:
//! it stays `Copy` like [`crate::DeviceCaps`] and a lookup cannot hit two configs.

/// The operand dtype of a matrix fragment. The accumulator is f32 on every config.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FragmentDtype {
    /// bfloat16: the f32 exponent range with 8 significand bits.
    Bf16,
    /// IEEE binary16: 11 significand bits, overflows above 65504.
    F16,
}

impl FragmentDtype {
    /// Every dtype a fragment can take, in [`MatrixUnits`] slot order.
    pub const ALL: [FragmentDtype; 2] = [FragmentDtype::Bf16, FragmentDtype::F16];

    const fn slot(self) -> usize {
        match self {
            FragmentDtype::Bf16 => 0,
            FragmentDtype::F16 => 1,
        }
    }
}

/// The `m x n x k` shape of one fragment multiply: `C[m, n] += A[m, k] * B[k, n]`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FragmentShape {
    /// Rows of A and C.
    pub m: u32,
    /// Columns of B and C.
    pub n: u32,
    /// Columns of A and rows of B.
    pub k: u32,
}

impl FragmentShape {
    /// The one shape any backend's codegen lowers today.
    pub const M16N16K16: FragmentShape = FragmentShape {
        m: 16,
        n: 16,
        k: 16,
    };
}

/// A hardware lane map a fragment's registers follow, documented by the vendor and verified on a device.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LaneMap {
    /// RDNA3 / RDNA3.5 (gfx11) WMMA, wave32: lane `l` holds A row and B column `l % 16`, the two half-waves
    /// replicate, and accumulator element `i` of lane `l` is row `2 i + l / 16`, column `l % 16`
    /// (`measure/wmma_probe.hip`, hardware-verified on gfx1151).
    Rdna3,
}

/// Whether a kernel can address the elements of a fragment.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FragmentLayout {
    /// The element-to-lane map is fixed by the hardware and known: a kernel reads and writes accumulator
    /// elements in place.
    Declared(LaneMap),
    /// The map is implementation-defined (`VK_KHR_cooperative_matrix`, the NVPTX `wmma` API): elements go
    /// through one store to shared memory and a reload.
    Opaque,
}

/// The error a fragment's narrow operand rounding costs against f32 semantics: round to nearest even of one
/// operand, f32 accumulation (design section 2.4, POOT-ADR-0116 applied to GPU matrix units).
///
/// The componentwise bound is `|y_narrow - y_exact| <= c * u * sum |a||b|` with `u` the dtype's unit roundoff
/// and `c = componentwise_coefficient_milli / 1000`. It is the measured worst case on the heavy-tailed bank
/// plus a margin, deliberately below the provable `c = 1` (a bound that cannot fail catches nothing) and below
/// what a truncating conversion measures (0.96). It covers the region classes and data it was measured on.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct OperandNarrowing {
    /// `c` in thousandths.
    pub componentwise_coefficient_milli: u32,
    /// The largest finite magnitude the dtype holds: an f32 operand beyond it narrows to infinity and the
    /// bound is void (the conversion does not saturate).
    pub max_magnitude: f32,
}

// `max_magnitude` is a finite constant of the dtype, never NaN, so equality is reflexive.
impl Eq for OperandNarrowing {}

impl OperandNarrowing {
    /// The contract of `dtype`'s rounding. Measured on gfx1151 (bf16 c = 0.63, f16 c = 0.47, recorded 0.75
    /// with a margin; `measure/gpu_run.log`). It is a property of the round-to-nearest conversion and the f32
    /// accumulation, not of one device, so every target reporting the dtype shares it; each device's own
    /// contract row re-measures it.
    pub const fn of(dtype: FragmentDtype) -> Self {
        match dtype {
            FragmentDtype::Bf16 => OperandNarrowing {
                componentwise_coefficient_milli: 750,
                max_magnitude: f32::from_bits(0x7f7f_0000),
            },
            FragmentDtype::F16 => OperandNarrowing {
                componentwise_coefficient_milli: 750,
                max_magnitude: 65_504.0,
            },
        }
    }
}

/// One matrix-multiply configuration a device offers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FragmentConfig {
    /// The dtype of A and B.
    pub dtype: FragmentDtype,
    /// The fragment multiply's shape.
    pub shape: FragmentShape,
    /// Whether elements are addressable.
    pub layout: FragmentLayout,
    /// Lanes per fragment collective (the subgroup the fragment is held across).
    pub subgroup: u32,
    /// What the operand rounding costs.
    pub contract: OperandNarrowing,
}

impl FragmentConfig {
    /// A 16x16x16 config of `dtype` held across `subgroup` lanes.
    pub const fn m16n16k16(dtype: FragmentDtype, layout: FragmentLayout, subgroup: u32) -> Self {
        FragmentConfig {
            dtype,
            shape: FragmentShape::M16N16K16,
            layout,
            subgroup,
            contract: OperandNarrowing::of(dtype),
        }
    }
}

/// One `VK_KHR_cooperative_matrix` configuration a Vulkan device reports (or wgpu's
/// `CooperativeMatrixProperties`, the same data), reduced to the facts [`MatrixUnits::cooperative_matrix`]
/// classifies. Runtimes build it from their API's own property type; subgroup scope is filtered by the caller.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CooperativeMatrixConfig {
    /// Matrix-multiply dimensions.
    pub shape: FragmentShape,
    /// The A/B operand type is 16-bit float.
    pub ab_f16: bool,
    /// The accumulator/result type is 32-bit float.
    pub result_f32: bool,
}

/// The matrix units of a device: empty when it has none poot's codegen drives.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct MatrixUnits {
    configs: [Option<FragmentConfig>; FragmentDtype::ALL.len()],
}

impl MatrixUnits {
    /// A device with no matrix unit the code generator targets.
    pub const NONE: MatrixUnits = MatrixUnits {
        configs: [None; FragmentDtype::ALL.len()],
    };

    /// Whether the device has no matrix unit.
    pub fn is_empty(&self) -> bool {
        self.configs.iter().all(Option::is_none)
    }

    /// The config for operands of `dtype`, if the device has one.
    pub fn config(&self, dtype: FragmentDtype) -> Option<&FragmentConfig> {
        self.configs[dtype.slot()].as_ref()
    }

    /// Every config, bf16 first.
    pub fn iter(&self) -> impl Iterator<Item = &FragmentConfig> {
        self.configs.iter().flatten()
    }

    /// The units of exactly `configs`. A later config of the same dtype replaces an earlier one.
    pub fn of(configs: impl IntoIterator<Item = FragmentConfig>) -> Self {
        let mut units = MatrixUnits::NONE;
        for config in configs {
            units.configs[config.dtype.slot()] = Some(config);
        }
        units
    }

    /// An AMD GPU by its gfx code: gfx11 (RDNA3 and 3.5) has WMMA 16x16x16 in bf16 and f16 with a known lane
    /// map across 32 lanes. gfx12 packs fragments differently and CDNA is MFMA; neither has a lane map or a
    /// lowering yet, and an unrecognized code is no unit rather than a guessed one.
    pub fn amd(gfx: &str, wave: u32) -> Self {
        let gfx11 = gfx.strip_prefix("gfx").is_some_and(|n| n.starts_with("11"));
        if !gfx11 || wave != 32 {
            return MatrixUnits::NONE;
        }
        Self::of(FragmentDtype::ALL.map(|dtype| {
            FragmentConfig::m16n16k16(dtype, FragmentLayout::Declared(LaneMap::Rdna3), 32)
        }))
    }

    /// An NVIDIA GPU by CUDA compute capability major: sm_80 and newer have `wmma` 16x16x16 in bf16 and f16,
    /// 32-lane warps. The `wmma` API's fragment layout is opaque.
    pub fn nvidia(compute_capability_major: u32) -> Self {
        if compute_capability_major < 8 {
            return MatrixUnits::NONE;
        }
        Self::of(
            FragmentDtype::ALL
                .map(|dtype| FragmentConfig::m16n16k16(dtype, FragmentLayout::Opaque, 32)),
        )
    }

    /// A Vulkan device from its reported cooperative-matrix configurations: the f16-in f32-out 16x16x16
    /// config is an `Opaque` f16 fragment held across `subgroup` lanes. The extension reports no bf16
    /// config poot lowers (Mesa disables `VK_KHR_shader_bfloat16` on gfx11), and any other shape or type is
    /// not a config the code generator has a body for.
    pub fn cooperative_matrix(
        reported: impl IntoIterator<Item = CooperativeMatrixConfig>,
        subgroup: u32,
    ) -> Self {
        Self::of(
            reported
                .into_iter()
                .filter(|c| c.shape == FragmentShape::M16N16K16 && c.ab_f16 && c.result_f32)
                .map(|_| {
                    FragmentConfig::m16n16k16(FragmentDtype::F16, FragmentLayout::Opaque, subgroup)
                }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dtypes(units: &MatrixUnits) -> Vec<FragmentDtype> {
        units.iter().map(|c| c.dtype).collect()
    }

    /// gfx11 reports bf16 and f16 with the declared RDNA3 lane map; gfx12, CDNA and an unknown code report
    /// none. Mutation: match `"12"` as well in `amd`; the gfx1201 row goes red.
    #[test]
    fn amd_gfx11_has_declared_bf16_and_f16_and_nothing_else_has_a_unit() {
        let units = MatrixUnits::amd("gfx1151", 32);
        assert_eq!(dtypes(&units), [FragmentDtype::Bf16, FragmentDtype::F16]);
        assert!(
            units
                .iter()
                .all(|c| c.layout == FragmentLayout::Declared(LaneMap::Rdna3) && c.subgroup == 32)
        );
        for gfx in ["gfx1201", "gfx90a", "gfx900", "gfx9"] {
            assert!(MatrixUnits::amd(gfx, 32).is_empty(), "{gfx}");
        }
        assert!(MatrixUnits::amd("gfx1151", 64).is_empty());
    }

    /// Only an f16-in f32-out 16x16x16 config is a unit; it is f16, opaque. Mutation: drop the `result_f32`
    /// filter; the f16-accumulate row goes red.
    #[test]
    fn a_cooperative_matrix_list_yields_an_opaque_f16_unit_for_the_one_lowered_config() {
        let f16_f32 = CooperativeMatrixConfig {
            shape: FragmentShape::M16N16K16,
            ab_f16: true,
            result_f32: true,
        };
        let units = MatrixUnits::cooperative_matrix([f16_f32], 32);
        assert_eq!(dtypes(&units), [FragmentDtype::F16]);
        assert_eq!(
            units.config(FragmentDtype::F16).unwrap().layout,
            FragmentLayout::Opaque
        );
        assert!(units.config(FragmentDtype::Bf16).is_none());

        for other in [
            CooperativeMatrixConfig {
                result_f32: false,
                ..f16_f32
            },
            CooperativeMatrixConfig {
                ab_f16: false,
                ..f16_f32
            },
            CooperativeMatrixConfig {
                shape: FragmentShape { m: 8, n: 8, k: 8 },
                ..f16_f32
            },
        ] {
            assert!(
                MatrixUnits::cooperative_matrix([other], 32).is_empty(),
                "{other:?}"
            );
        }
        assert!(MatrixUnits::cooperative_matrix([], 32).is_empty());
    }

    /// sm_80 and newer have opaque bf16 and f16; older has none. Mutation: change the threshold to 9.
    #[test]
    fn nvidia_sm80_and_newer_have_opaque_bf16_and_f16() {
        assert_eq!(
            dtypes(&MatrixUnits::nvidia(8)),
            [FragmentDtype::Bf16, FragmentDtype::F16]
        );
        assert!(MatrixUnits::nvidia(7).is_empty());
    }

    /// The conversion saturates nowhere: the recorded range is the dtype's largest finite value.
    #[test]
    fn the_contract_range_is_the_dtypes_largest_finite_value() {
        assert_eq!(
            OperandNarrowing::of(FragmentDtype::F16).max_magnitude,
            65_504.0
        );
        let bf16_max = OperandNarrowing::of(FragmentDtype::Bf16).max_magnitude;
        assert!(bf16_max.is_finite() && bf16_max > 3.0e38 && bf16_max < f32::MAX);
    }
}
