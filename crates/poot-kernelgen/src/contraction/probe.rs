//! The register probe of a candidate tiling (POOT-1505): the body the generator emits for a contraction's shape under
//! one tiling, its operands, weight format and rows as the shape states them, with no epilogue. A device's compiler
//! compiles it to say whether the tiling spills ([`Feasibility::Compiled`]); every place the selection asks (the
//! precision pass, the expert grouping, the final plan) asks of this one body, so they cannot disagree.

use poot_graph_ir::{FragmentDtype, OperandPrecision};
use poot_kernel_ir::Body;
use poot_target::{BufferStorage, RoundingConversions};

use super::{DenseTiledWeight, RegisterTiling, StagedWeight, TiledExtents, TiledRows, TiledShape};
use crate::KernelGenError;
use crate::helpers::WeightLayout;

/// A device compiler's answer for a probe: whether the compiled body keeps its values in registers (no local memory).
/// Implemented by the planner's compile feedback.
pub trait SpillCheck {
    /// Whether `probe` compiles without spilling; `false` when it spills or does not compile.
    fn holds(&self, probe: &Body) -> bool;
}

/// What decides whether the device runs a candidate tiling's registers.
#[derive(Clone, Copy)]
pub enum Feasibility<'a> {
    /// The register model alone: what a tiling holds across its loop within the held share of the lane's registers
    /// (`poot_target::RegisterBudget::loop_registers`).
    Model,
    /// The device's compiler: the candidate's probe ([`probe_body`]) compiles without spilling. The register model
    /// only prunes what the lane's registers cannot hold at all (`poot_target::RegisterBudget::lane_registers`).
    Compiled(&'a dyn SpillCheck),
}

impl std::fmt::Debug for Feasibility<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model => f.write_str("Model"),
            Self::Compiled(_) => f.write_str("Compiled"),
        }
    }
}

/// The precision `shape`'s terms stage: exact, narrowed, or split (two tiles of an operand) to their fragment dtype.
fn precision_of(shape: TiledShape) -> OperandPrecision {
    match shape.terms.fragment {
        None => OperandPrecision::Exact,
        Some(dtype) if shape.terms.activation > 1 || shape.terms.weight > 1 => {
            OperandPrecision::Split(dtype)
        }
        Some(dtype) => OperandPrecision::Narrow(dtype),
    }
}

/// The body the generator emits for `shape` under `tiling`, with no epilogue (POOT-1505): a packed weight's
/// contraction (grouped rows: the expert-indexed one), or a dense weight's (grouped rows: the indexed contraction over
/// a stacked weight). A 16-bit dense weight is staged from bf16 words, a 32-bit one from f32; a 16-bit activation is
/// stored in the shape's fragment dtype (f16 when it has none). The metadata's values are runtime arguments, not part
/// of what the compiler allocates.
pub fn probe_body(
    shape: TiledShape,
    tiling: RegisterTiling,
    conversions: RoundingConversions,
) -> Result<Body, KernelGenError> {
    let precision = precision_of(shape);
    let schedule = tiling.schedule();
    let (k, n) = (shape.k as u32, shape.out_per_block as u32);
    let name = format!(
        "register_probe_{:?}_{:?}_{k}x{n}_{:?}",
        shape.weight, precision, schedule
    )
    .replace(|c: char| !c.is_ascii_alphanumeric(), "_");
    let grouped = matches!(shape.rows, TiledRows::Grouped { .. });
    let experts = match shape.rows {
        TiledRows::Grouped { groups, .. } => groups.saturating_sub(1) as u32,
        TiledRows::Dense => 0,
    };
    match shape.weight {
        StagedWeight::Packed(format) => {
            let meta: Vec<u32> = if grouped {
                vec![k, n, experts, 0, 0, 0, 0]
            } else {
                vec![k, n]
            };
            crate::packed::packed_contraction(
                &name,
                format,
                schedule,
                &meta,
                None,
                if grouped {
                    crate::PackedRows::Indexed
                } else {
                    crate::PackedRows::Blocks
                },
                precision,
                conversions,
            )
        }
        StagedWeight::Dense { bits } => {
            let storage = if bits == 16 {
                BufferStorage::bf16_packed()
            } else {
                BufferStorage::f32()
            };
            if grouped {
                super::indexed_contraction(
                    &name,
                    schedule,
                    shape.layout,
                    storage,
                    &[k, n, 0, 0, 0, experts],
                    None,
                    precision,
                    conversions,
                )
            } else {
                let weight = match shape.layout {
                    WeightLayout::Nk => DenseTiledWeight::Nk(storage),
                    WeightLayout::Kn => DenseTiledWeight::KnF32,
                };
                super::dense_tiled_contraction(
                    &name,
                    weight,
                    schedule,
                    TiledExtents {
                        k: shape.k,
                        out_per_block: shape.out_per_block,
                    },
                    None,
                    super::TiledRead {
                        precision,
                        activation_stored: (shape.activation_bits == 16)
                            .then(|| shape.terms.fragment.unwrap_or(FragmentDtype::F16)),
                        conversions,
                    },
                )
            }
        }
    }
}
