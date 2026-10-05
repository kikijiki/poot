//! Card 542a's descriptor-driven packed kernel lowering table: which generated op a packed weight's
//! format admits, on which backend.

use poot_quant::PackedWeight;
use poot_target::Backend;

// --- Card 542a: the descriptor-driven packed kernel lowering table -----------------------------

/// A generated packed kernel's op, without the [`poot_kernelgen::Schedule`]/[`poot_kernelgen::RowSelect`]
/// detail a lowering-table row does not admit or refuse on (dquant.md D3, section 5): the schedule is
/// chosen from shapes, never from the format, so it cannot change whether a row admits a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackedKernelOpKind {
    Materialize,
    RowGather,
    Contraction,
}

impl From<poot_kernelgen::PackedKernelOp> for PackedKernelOpKind {
    fn from(op: poot_kernelgen::PackedKernelOp) -> Self {
        match op {
            poot_kernelgen::PackedKernelOp::Materialize => Self::Materialize,
            poot_kernelgen::PackedKernelOp::RowGather => Self::RowGather,
            poot_kernelgen::PackedKernelOp::Contraction { .. } => Self::Contraction,
        }
    }
}

/// The three lowering backends, without [`AmdArch`](poot_target::AmdArch)'s runtime gfx/wave detail:
/// packed-lowering admission never depends on which AMD part is attached, only on the backend
/// family, and a bare discriminant is `const`-constructible (a full `Backend` is not: `AmdArch` is
/// detected at runtime, never a compile-time constant).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackedLoweringBackend {
    SpirvVulkan,
    AmdGcn,
    Nvptx,
}

impl From<Backend> for PackedLoweringBackend {
    fn from(backend: Backend) -> Self {
        match backend {
            Backend::SpirvVulkan => Self::SpirvVulkan,
            Backend::AmdGcn(_) => Self::AmdGcn,
            Backend::Nvptx => Self::Nvptx,
        }
    }
}

/// One row of the descriptor-driven packed lowering table (dquant.md D3, section 5). `admits` is a
/// descriptor PREDICATE, never a list of scheme names (the 539a rule applied to planning): a new
/// format that satisfies an existing predicate needs no new row.
pub struct PackedLoweringRow {
    pub op: PackedKernelOpKind,
    pub backends: &'static [PackedLoweringBackend],
    pub admits: fn(&poot_quant::format::FormatDescriptor) -> bool,
}

/// Every block-storage descriptor (dquant.md D3's staging predicates): card 542a's block-32 formats
/// (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `IQ4_NL`, MXFP4, no `SubScale`/`SubMin`) plus 542b's
/// K-quant super-blocks (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`, packed `SubScale`/`SubMin`
/// sub-factors): `DecodePlan` (`poot-kernelgen`) reads both shapes off the same descriptor fields, so
/// admission never distinguishes them.
fn admits_blocks(descriptor: &poot_quant::format::FormatDescriptor) -> bool {
    matches!(descriptor.storage, poot_quant::format::Storage::Blocks(_))
}

/// Every planar-storage descriptor (card 542c, dquant.md D3/D4's staging predicate): the six E4M3
/// per-channel/128x128 scale encodings, E2M1 row-32, GPTQ (contiguous and act-order) and AWQ.
/// `poot-kernelgen`'s `PlanarPlan` reads the same descriptor fields for every one of them, so
/// admission is the storage shape, never a format list (`admits_blocks`'s doc, restated for the
/// other front end).
fn admits_planar(descriptor: &poot_quant::format::FormatDescriptor) -> bool {
    matches!(descriptor.storage, poot_quant::format::Storage::Planar(_))
}

/// `admits_blocks(d) || admits_planar(d)`: every descriptor `poot-kernelgen`'s two front ends cover
/// (dquant.md D4, "two addressing front ends... come from `Storage`"). `Materialize` and
/// `Contraction` admit both storages; `RowGather` stays block-only (`admits_blocks`) - its one
/// production consumer is GGUF token-embedding gather, and no planar format's row is gathered today
/// (`poot_kernelgen::packed::row_gather`'s explicit refusal for `Storage::Planar`).
fn admits_blocks_or_planar(descriptor: &poot_quant::format::FormatDescriptor) -> bool {
    admits_blocks(descriptor) || admits_planar(descriptor)
}

const EVERY_BACKEND: &[PackedLoweringBackend] = &[
    PackedLoweringBackend::SpirvVulkan,
    PackedLoweringBackend::AmdGcn,
    PackedLoweringBackend::Nvptx,
];

/// The descriptor-driven packed kernel lowering table (card 542a covers `Materialize`, `RowGather`
/// and `Contraction` for the block-32 formats on every backend; 542b widens block admission to every
/// block-storage descriptor; 542c adds the planar-storage predicate for `Materialize`/`Contraction`).
pub const PACKED_LOWERING: &[PackedLoweringRow] = &[
    PackedLoweringRow {
        op: PackedKernelOpKind::Materialize,
        backends: EVERY_BACKEND,
        admits: admits_blocks_or_planar,
    },
    PackedLoweringRow {
        op: PackedKernelOpKind::RowGather,
        backends: EVERY_BACKEND,
        admits: admits_blocks,
    },
    PackedLoweringRow {
        op: PackedKernelOpKind::Contraction,
        backends: EVERY_BACKEND,
        admits: admits_blocks_or_planar,
    },
];

/// Select a generated packed kernel from `table`: `op`'s [`PackedKernelOpKind`] must have a row
/// admitting `weight`'s descriptor on `backend`, or the equation is refused by name
/// (`Capability::PackedLowering`) at claim time, never a panic (dquant.md section 5, SC-006).
pub fn select_packed_kernel(
    table: &[PackedLoweringRow],
    weight: PackedWeight,
    op: poot_kernelgen::PackedKernelOp,
    backend: Backend,
) -> Result<poot_kernelgen::PackedKernelSpec, crate::refusal::Capability> {
    let format = weight.format();
    let descriptor = format.descriptor();
    let kind = PackedKernelOpKind::from(op);
    let backend_kind = PackedLoweringBackend::from(backend);
    let admitted = table.iter().any(|row| {
        row.op == kind && row.backends.contains(&backend_kind) && (row.admits)(&descriptor)
    });
    if !admitted {
        return Err(crate::refusal::Capability::PackedLowering { format, op: kind });
    }
    let spec = poot_kernelgen::PackedKernelSpec { format, op };
    // The table's `admits` predicate is a descriptor property (dquant.md D3): it says this format's
    // shape is card 542a's staging, not that `packed_kernel` can build every body for it (a field of
    // more than two pieces, for one, `DecodePlan::of` also rejects). Building the body here, at claim
    // time, catches the generator's own precondition the same way card 531b's `Capability::KernelGen`
    // already does for the other generator families, rather than deferring the same failure to a
    // panic inside compile.
    poot_kernelgen::packed_kernel("select_packed_kernel_probe", spec)
        .map_err(crate::refusal::Capability::KernelGen)?;
    Ok(spec)
}

#[cfg(test)]
mod packed_lowering_tests {
    use poot_kernelgen::{PackedKernelOp, RowSelect, Schedule};
    use poot_quant::format::{GroupMap, ScaleEncoding, WeightFormat};

    use super::*;
    use crate::refusal::Capability;

    /// SC-006 (card 542a): a test-only table without the Gemv row yields
    /// `Refusal { missing: PackedLowering { format, op } }` naming the equation at claim time, not a
    /// panic. Mutation: restore the row (use `PACKED_LOWERING`, which does have every op); the
    /// refusal row goes red (the call succeeds instead of refusing).
    #[test]
    fn a_table_without_the_gemv_row_refuses_by_name_not_a_panic() {
        let table = &[
            PackedLoweringRow {
                op: PackedKernelOpKind::Materialize,
                backends: &[PackedLoweringBackend::SpirvVulkan],
                admits: admits_blocks,
            },
            PackedLoweringRow {
                op: PackedKernelOpKind::RowGather,
                backends: &[PackedLoweringBackend::SpirvVulkan],
                admits: admits_blocks,
            },
            // No Contraction row: a Gemv contraction has nothing to admit it.
        ];
        let weight = PackedWeight::try_new(WeightFormat::Q4_0, [4, 32]).unwrap();
        let op = PackedKernelOp::Contraction {
            rows: RowSelect::Dense,
            schedule: Schedule::Gemv {
                width: 256,
                cols: 64,
                unroll: 4,
            },
        };
        assert_eq!(
            select_packed_kernel(table, weight, op, Backend::SpirvVulkan),
            Err(Capability::PackedLowering {
                format: WeightFormat::Q4_0,
                op: PackedKernelOpKind::Contraction,
            })
        );
        // Restoring the row (the real table) makes the same call succeed.
        assert_eq!(
            select_packed_kernel(PACKED_LOWERING, weight, op, Backend::SpirvVulkan),
            Ok(poot_kernelgen::PackedKernelSpec {
                format: WeightFormat::Q4_0,
                op
            })
        );
    }

    /// Card 542b: `admits_blocks` widens 542a's block-32-only predicate to every block-storage
    /// descriptor, so a K-quant super-block format (`SubScale`/`SubMin` sub-factors) is now admitted
    /// against the real table on every backend and op, the opposite of 542a's own
    /// `the_real_table_refuses_a_format_admits_blocks_declines` (superseded: it asserted the pre-542b
    /// refusal this row now proves lifted).
    #[test]
    fn the_real_table_now_admits_a_kquant_format_on_every_backend() {
        let weight = PackedWeight::try_new(WeightFormat::Q4_K, [4, 256]).unwrap();
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
        ] {
            assert_eq!(
                select_packed_kernel(
                    PACKED_LOWERING,
                    weight,
                    PackedKernelOp::Materialize,
                    backend
                ),
                Ok(poot_kernelgen::PackedKernelSpec {
                    format: WeightFormat::Q4_K,
                    op: PackedKernelOp::Materialize,
                }),
                "{backend:?}: Q4_K (SubScale/SubMin) must now be admitted (card 542b)"
            );
        }
    }

    /// SC-003 (card 542b): a test-only table with the Tiled row removed yields a typed `Refusal`
    /// naming the equation, not a panic, as 542a's SC-006. Mutation: restore the row (use
    /// `PACKED_LOWERING`, which does admit `Contraction` on every backend); the refusal row goes red
    /// (the call succeeds instead of refusing).
    #[test]
    fn a_table_without_the_contraction_row_refuses_a_tiled_claim_by_name_not_a_panic() {
        let table = &[
            PackedLoweringRow {
                op: PackedKernelOpKind::Materialize,
                backends: &[PackedLoweringBackend::SpirvVulkan],
                admits: admits_blocks,
            },
            PackedLoweringRow {
                op: PackedKernelOpKind::RowGather,
                backends: &[PackedLoweringBackend::SpirvVulkan],
                admits: admits_blocks,
            },
            // No Contraction row: a Tiled contraction has nothing to admit it.
        ];
        let weight = PackedWeight::try_new(WeightFormat::Q4_K, [4, 256]).unwrap();
        let op = PackedKernelOp::Contraction {
            rows: RowSelect::Dense,
            schedule: Schedule::Tiled {
                tile: poot_kernelgen::TileSize::new(8).expect("a dense contraction tiles"),
            },
        };
        assert_eq!(
            select_packed_kernel(table, weight, op, Backend::SpirvVulkan),
            Err(Capability::PackedLowering {
                format: WeightFormat::Q4_K,
                op: PackedKernelOpKind::Contraction,
            })
        );
        // Restoring the row (the real table) makes the same call succeed.
        assert_eq!(
            select_packed_kernel(PACKED_LOWERING, weight, op, Backend::SpirvVulkan),
            Ok(poot_kernelgen::PackedKernelSpec {
                format: WeightFormat::Q4_K,
                op
            })
        );
    }

    /// Card 542c: `admits_planar` widens the table to every planar-storage descriptor, so a GPTQ
    /// format (three sources: `Codes`, `Zero`, `Scale`) is now admitted for `Materialize` and
    /// `Contraction` on every backend - the planar counterpart of
    /// `the_real_table_now_admits_a_kquant_format_on_every_backend`.
    #[test]
    fn the_real_table_now_admits_a_planar_format_on_every_backend() {
        let format = WeightFormat::Gptq {
            groups: GroupMap::Contiguous {
                size: std::num::NonZeroUsize::new(8).unwrap(),
            },
        };
        let weight = PackedWeight::try_new(format, [4, 16]).unwrap();
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
        ] {
            assert_eq!(
                select_packed_kernel(
                    PACKED_LOWERING,
                    weight,
                    PackedKernelOp::Materialize,
                    backend
                ),
                Ok(poot_kernelgen::PackedKernelSpec {
                    format,
                    op: PackedKernelOp::Materialize,
                }),
                "{backend:?}: GPTQ Materialize must be admitted (card 542c)"
            );
            let op = PackedKernelOp::Contraction {
                rows: RowSelect::Dense,
                schedule: Schedule::Gemv {
                    width: 256,
                    cols: 8,
                    unroll: 8,
                },
            };
            assert_eq!(
                select_packed_kernel(PACKED_LOWERING, weight, op, backend),
                Ok(poot_kernelgen::PackedKernelSpec { format, op }),
                "{backend:?}: GPTQ Contraction must be admitted (card 542c)"
            );
        }
    }

    /// Card 542c: `RowGather` stays block-only - no production consumer gathers a planar-format
    /// row, and `poot_kernelgen::packed::row_gather` itself refuses `Storage::Planar` (the emitter's
    /// own precondition this table's `RowGather` row must never route past).
    #[test]
    fn the_real_table_does_not_admit_a_planar_format_for_row_gather() {
        let format = WeightFormat::E2m1Row32;
        let weight = PackedWeight::try_new(format, [4, 64]).unwrap();
        assert_eq!(
            select_packed_kernel(
                PACKED_LOWERING,
                weight,
                PackedKernelOp::RowGather,
                Backend::SpirvVulkan
            ),
            Err(Capability::PackedLowering {
                format,
                op: PackedKernelOpKind::RowGather,
            })
        );
    }

    /// Card 542d, ADR-0104: whether a format executes is decided here, at planning, never by a
    /// loader-side registry (`poot_load::packed_safetensors::validation`'s
    /// `expected_dtype_derives_from_the_descriptor_for_an_unregistered_planar_format` proves the
    /// loader-side half: `E4m3PerChannel` - never one of the four `PackedLinearFormat` cells card
    /// 542d deleted - loads with no panic). This is the planning-side half of the same story: the
    /// same unregistered-but-described format has a descriptor (so `admits_planar` would admit it
    /// for `Materialize`/`Contraction`), but `RowGather` stays block-only, so it is refused here by
    /// name, a typed `Refusal`, not a panic - the same shape as
    /// `the_real_table_does_not_admit_a_planar_format_for_row_gather` above, for a format that was
    /// never registered at all.
    #[test]
    fn an_unregistered_planar_format_is_refused_at_planning_by_name_not_a_panic() {
        let format = WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::F32,
        };
        let weight = PackedWeight::try_new(format, [4, 8]).unwrap();
        assert_eq!(
            select_packed_kernel(
                PACKED_LOWERING,
                weight,
                PackedKernelOp::RowGather,
                Backend::SpirvVulkan
            ),
            Err(Capability::PackedLowering {
                format,
                op: PackedKernelOpKind::RowGather,
            })
        );
        // The same format IS admitted for Materialize (ADR-0104: the loader never gated on
        // registration, and neither does planning admission - `admits_planar` is a storage-shape
        // predicate, not a format list).
        assert_eq!(
            select_packed_kernel(
                PACKED_LOWERING,
                weight,
                PackedKernelOp::Materialize,
                Backend::SpirvVulkan
            ),
            Ok(poot_kernelgen::PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            })
        );
    }

    /// SC-003 (card 542c): a test-only table with the planar row removed yields a typed `Refusal`
    /// naming the equation, not a panic, as 542a's SC-006 and 542b's SC-003. Mutation: restore the
    /// row (use `PACKED_LOWERING`, which does admit `Materialize` for planar storage); the refusal
    /// row goes red (the call succeeds instead of refusing).
    #[test]
    fn a_table_without_the_planar_row_refuses_a_planar_materialize_by_name_not_a_panic() {
        let table = &[
            PackedLoweringRow {
                op: PackedKernelOpKind::Materialize,
                backends: &[PackedLoweringBackend::SpirvVulkan],
                admits: admits_blocks,
            },
            PackedLoweringRow {
                op: PackedKernelOpKind::RowGather,
                backends: &[PackedLoweringBackend::SpirvVulkan],
                admits: admits_blocks,
            },
            PackedLoweringRow {
                op: PackedKernelOpKind::Contraction,
                backends: &[PackedLoweringBackend::SpirvVulkan],
                admits: admits_blocks,
            },
            // Every row admits block storage only: a planar descriptor has nothing to admit it.
        ];
        let format = WeightFormat::Awq {
            group_size: std::num::NonZeroUsize::new(8).unwrap(),
        };
        let weight = PackedWeight::try_new(format, [8, 16]).unwrap();
        assert_eq!(
            select_packed_kernel(
                table,
                weight,
                PackedKernelOp::Materialize,
                Backend::SpirvVulkan
            ),
            Err(Capability::PackedLowering {
                format,
                op: PackedKernelOpKind::Materialize,
            })
        );
        // Restoring the row (the real table) makes the same call succeed.
        assert_eq!(
            select_packed_kernel(
                PACKED_LOWERING,
                weight,
                PackedKernelOp::Materialize,
                Backend::SpirvVulkan
            ),
            Ok(poot_kernelgen::PackedKernelSpec {
                format,
                op: PackedKernelOp::Materialize,
            })
        );
    }
}
