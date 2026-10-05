//! The graph constant names and source types of a packed weight.
//!
//! A packed weight enters a graph as one `I8` source constant per [`poot_quant::SourceRole`]
//! [`poot_quant::PackedWeight::sources`] names, each named after the linear it belongs to. This
//! module owns that spelling and the type each constant is staged with. Builders
//! ([`crate::ops::packed_linear`]), planners, the CPU oracle, loaders, and model adapters all go
//! through it, so the suffix and the source byte width exist in exactly one place.

use std::fmt;

use poot_quant::{OperandRole, PackedWeight, SourceRole};

use crate::types::{DType, TensorType};

/// Every [`SourceRole`] a packed source name can carry.
const ALL_ROLES: [SourceRole; 8] = [
    SourceRole::Blocks,
    SourceRole::Planar(OperandRole::Codes),
    SourceRole::Planar(OperandRole::Scale),
    SourceRole::Planar(OperandRole::SubScale),
    SourceRole::Planar(OperandRole::Zero),
    SourceRole::Planar(OperandRole::Min),
    SourceRole::Planar(OperandRole::SubMin),
    SourceRole::Planar(OperandRole::GroupIndex),
];

/// The graph constant name of one packed weight's source component:
/// `{linear_id}.packed_weight_source`, `{linear_id}.packed_scale_source`, and so on, one suffix per
/// [`SourceRole`].
///
/// No constructor accepts a whole name, so a caller cannot hand-spell the suffix and get this type;
/// build one from a linear id with [`PackedSourceName::new`], or recover a staged one with
/// [`PackedSourceName::parse`].
///
/// The split is lexical. An empty linear id round-trips like any other; graph construction rejects
/// it separately, before a name is staged.
///
/// ```
/// use poot_graph_ir::PackedSourceName;
///
/// let staged = PackedSourceName::weight("model.layers.0.mlp.down_proj");
/// let parsed = PackedSourceName::parse(staged.as_str()).unwrap();
/// assert_eq!(parsed.linear_id(), "model.layers.0.mlp.down_proj");
/// assert_eq!(parsed, staged);
/// ```
///
/// An interface that asks for a source name asks for this type, so a re-spelled literal does not
/// type-check:
///
/// ```compile_fail,E0308
/// use poot_graph_ir::PackedSourceName;
///
/// fn bind_source(_: &PackedSourceName) {}
/// bind_source("model.layers.0.mlp.down_proj.packed_weight_source");
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PackedSourceName {
    text: String,
    linear_id_len: usize,
    role: SourceRole,
}

impl PackedSourceName {
    /// The dotted suffix `role` appends to a linear id. This is the one spelling; error messages
    /// that quote it read it from here.
    pub const fn suffix(role: SourceRole) -> &'static str {
        match role {
            SourceRole::Blocks => ".packed_blocks_source",
            SourceRole::Planar(OperandRole::Codes) => ".packed_weight_source",
            SourceRole::Planar(OperandRole::Scale) => ".packed_scale_source",
            SourceRole::Planar(OperandRole::SubScale) => ".packed_subscale_source",
            SourceRole::Planar(OperandRole::Zero) => ".packed_zero_source",
            SourceRole::Planar(OperandRole::Min) => ".packed_min_source",
            SourceRole::Planar(OperandRole::SubMin) => ".packed_submin_source",
            SourceRole::Planar(OperandRole::GroupIndex) => ".packed_group_index_source",
        }
    }

    pub fn new(linear_id: &str, role: SourceRole) -> Self {
        Self {
            text: format!("{linear_id}{}", Self::suffix(role)),
            linear_id_len: linear_id.len(),
            role,
        }
    }

    pub fn weight(linear_id: &str) -> Self {
        Self::new(linear_id, SourceRole::Planar(OperandRole::Codes))
    }

    pub fn scale(linear_id: &str) -> Self {
        Self::new(linear_id, SourceRole::Planar(OperandRole::Scale))
    }

    /// Both source names of one linear built from exactly `Codes` and `Scale`, weight first.
    pub fn pair(linear_id: &str) -> [Self; 2] {
        [Self::weight(linear_id), Self::scale(linear_id)]
    }

    /// Recover the linear id and role behind a staged constant name, or `None` when the name is not
    /// a packed source name.
    pub fn parse(name: &str) -> Option<Self> {
        ALL_ROLES.into_iter().find_map(|role| {
            let linear_id_len = name.strip_suffix(Self::suffix(role))?.len();
            Some(Self {
                text: name.to_string(),
                linear_id_len,
                role,
            })
        })
    }

    pub const fn role(&self) -> SourceRole {
        self.role
    }

    pub fn linear_id(&self) -> &str {
        &self.text[..self.linear_id_len]
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }
}

impl fmt::Display for PackedSourceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

impl AsRef<str> for PackedSourceName {
    fn as_ref(&self) -> &str {
        &self.text
    }
}

impl From<PackedSourceName> for String {
    fn from(name: PackedSourceName) -> Self {
        name.text
    }
}

/// The `I8` constant type `role` is staged with: `weight`'s source shape for that role
/// ([`PackedWeight::source_shape`]), never a decoded value.
pub fn packed_source_type(weight: PackedWeight, role: SourceRole) -> TensorType {
    TensorType::new(weight.source_shape(role).to_vec(), DType::I8)
}

/// Every source constant of one packed weight, in [`PackedWeight::sources`] order: the name to
/// stage and the type to stage it with. One constant per source - the registered E4M3/E2M1 cells
/// stage exactly two (weight, scale); other schemes stage as many as their descriptor names.
pub fn packed_source_constants(
    linear_id: &str,
    weight: PackedWeight,
) -> Vec<(PackedSourceName, TensorType)> {
    weight
        .sources()
        .into_iter()
        .map(|role| {
            let name = PackedSourceName::new(linear_id, role);
            let tensor_type = packed_source_type(weight, role);
            (name, tensor_type)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use poot_quant::format::{ScaleEncoding, WeightFormat};

    use super::*;

    /// The only place the graph spelling is written as a literal. Every other packed test follows
    /// [`PackedSourceName`], so changing the suffix turns this test red and nothing else.
    #[test]
    fn packed_source_names_use_the_pinned_spelling() {
        let weight = PackedSourceName::weight("model.layers.0.mlp.down_proj");
        let scale = PackedSourceName::scale("model.layers.0.mlp.down_proj");
        assert_eq!(
            weight.as_str(),
            "model.layers.0.mlp.down_proj.packed_weight_source"
        );
        assert_eq!(
            scale.as_str(),
            "model.layers.0.mlp.down_proj.packed_scale_source"
        );
        assert_eq!(weight.to_string(), weight.as_str());
        assert_eq!(String::from(scale.clone()), scale.as_str());
        assert_eq!(
            PackedSourceName::pair("model.layers.0.mlp.down_proj"),
            [weight, scale]
        );
    }

    #[test]
    fn packed_source_names_round_trip_through_parse() {
        for linear_id in ["", "layer", "model.layers.7.self_attn.q_proj", "a.b.c"] {
            for role in ALL_ROLES {
                let name = PackedSourceName::new(linear_id, role);
                assert_eq!(name.linear_id(), linear_id);
                assert_eq!(name.role(), role);
                let parsed = PackedSourceName::parse(name.as_str()).expect("staged name parses");
                assert_eq!(parsed, name);
                assert_eq!(parsed.linear_id(), linear_id);
                assert_eq!(parsed.role(), role);
            }
        }
    }

    #[test]
    fn parse_rejects_names_that_are_not_packed_sources() {
        for name in [
            "",
            "layer",
            "layer.weight",
            "layer.packed_weight_source.extra",
            "layer.packed_weight",
            "packed_weight_source",
        ] {
            assert_eq!(PackedSourceName::parse(name), None, "{name}");
        }
    }

    /// The staged types carry source bytes: their element counts are exactly the owner's buffer
    /// lengths, for every registered format.
    #[test]
    fn packed_source_types_span_the_authoritative_source_buffers() {
        let formats = [
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::Bf16,
            },
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::F32,
            },
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::E8m0,
            },
            WeightFormat::E2m1Row32,
        ];
        for (index, format) in formats.into_iter().enumerate() {
            let logical = if index % 2 == 0 { [2, 256] } else { [3, 129] };
            let weight = PackedWeight::try_new(format, logical).unwrap();
            let constants = packed_source_constants("layer", weight);
            let [(weight_name, weight_type), (scale_name, scale_type)] =
                <[(PackedSourceName, TensorType); 2]>::try_from(constants)
                    .expect("a registered packed-linear format has exactly two sources");

            assert_eq!(weight_name.role(), SourceRole::Planar(OperandRole::Codes));
            assert_eq!(scale_name.role(), SourceRole::Planar(OperandRole::Scale));
            assert_eq!(weight_type.dtype, DType::I8, "{format:?}");
            assert_eq!(scale_type.dtype, DType::I8, "{format:?}");
            assert_eq!(
                weight_type.shape,
                weight
                    .source_shape(SourceRole::Planar(OperandRole::Codes))
                    .to_vec(),
                "{format:?}"
            );
            assert_eq!(
                weight_type.shape.iter().product::<usize>(),
                weight.source_bytes(SourceRole::Planar(OperandRole::Codes)),
                "{format:?}"
            );
            assert_eq!(
                scale_type.shape.iter().product::<usize>(),
                weight.source_bytes(SourceRole::Planar(OperandRole::Scale)),
                "{format:?}"
            );
            assert_eq!(
                scale_type.shape[0],
                weight.source_shape(SourceRole::Planar(OperandRole::Scale))[0],
                "{format:?}"
            );
        }
    }
}
