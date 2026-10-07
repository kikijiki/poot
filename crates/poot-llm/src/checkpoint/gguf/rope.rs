use poot_graph_ir::rope_table::{RopeFlavor, RopeTables};
use poot_load::RopeScaling;
use poot_models::qwen2::Qwen2Config;
use poot_tensor::HostTensor;

/// The [`RopeFlavor`] an HF `rope_scaling` names. YaRN's omitted `beta_fast`/`beta_slow` take HF's
/// defaults (32 and 1); an unknown or absent type is plain RoPE.
fn rope_flavor(scaling: Option<&RopeScaling>, max_positions: usize) -> RopeFlavor<'_> {
    let Some(s) = scaling else {
        return RopeFlavor::Plain;
    };
    let original = s.original_max_position_embeddings;
    match s.rope_type.as_str() {
        "linear" => RopeFlavor::Linear { factor: s.factor },
        "dynamic" => RopeFlavor::DynamicNtk {
            factor: s.factor,
            original,
        },
        "yarn" => RopeFlavor::Yarn {
            factor: s.factor,
            original,
            beta_fast: s.beta_fast.unwrap_or(32.0),
            beta_slow: s.beta_slow.unwrap_or(1.0),
            attention_factor: s.attention_factor,
        },
        "llama3" => RopeFlavor::Llama3 {
            factor: s.factor,
            low_freq_factor: s.low_freq_factor,
            high_freq_factor: s.high_freq_factor,
            original,
        },
        "longrope" => RopeFlavor::LongRope {
            original,
            max_positions,
            short: s.short_factor.as_deref(),
            long: s.long_factor.as_deref(),
        },
        _ => RopeFlavor::Plain,
    }
}

/// The Runner's `rope.cos`/`rope.sin` tables, `[max_pos, rotary_dim]`, from
/// [`poot_graph_ir::rope_table::rope_tables`]: the table's capacity is the model's whole context.
pub(crate) fn rope_tables(
    cfg: &Qwen2Config,
    theta: f32,
    scaling: Option<&RopeScaling>,
    freq_factors: Option<&[f32]>,
) -> (HostTensor, HostTensor) {
    let (d, p) = (cfg.rotary_dim, cfg.max_pos);
    let RopeTables { cos, sin } =
        poot_graph_ir::rope_table::rope_tables(d, p, theta, &rope_flavor(scaling, p), freq_factors);
    (
        HostTensor::f32(vec![p, d], cos),
        HostTensor::f32(vec![p, d], sin),
    )
}
