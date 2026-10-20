//! The synthetic checkpoints of the shipped families, keyed by [`FamilyKey`].
//!
//! A family's [`crate::registry::FamilyEntry`] is its keys and its builder; its tiny deterministic
//! checkpoints live here, in test support, so a production build compiles none of them. Every
//! shipped family has one row: the registry tests fail on a family without one, and on a row for a
//! family that is not shipped.

use crate::architectures::{
    bert, bloom, deepseek2, deepseek3, deepseek4, deepseek32, gemma2, gemma3, gemma4, glm53_flash,
    gpt_oss, granite, llama, minimax_m2, mpt, nemotron_h, olmo2, olmoe, phi3, qwen2, qwen3,
    qwen3moe, qwen3next, qwen38, qwen38_27b, smollm3,
};
use crate::model::FamilyKey;
use crate::registry::Fixture;

/// A named extra fixture of a family: `(option name, fixture)`.
pub type FixtureVariant = (&'static str, fn() -> Fixture);

/// One family's checkpoints.
#[derive(Clone, Copy, Debug)]
pub struct FamilyFixtures {
    /// The family the checkpoints belong to.
    pub family: FamilyKey,
    /// The family's base checkpoint.
    pub base: fn() -> Fixture,
    /// More checkpoints of the same family, one per option a key can select (llama's Mixtral
    /// experts, granite's `granitemoe` experts), by name.
    pub variants: &'static [FixtureVariant],
}

const fn row(
    family: FamilyKey,
    base: fn() -> Fixture,
    variants: &'static [FixtureVariant],
) -> FamilyFixtures {
    FamilyFixtures {
        family,
        base,
        variants,
    }
}

/// Every shipped family's checkpoints, in the order of [`crate::registry::SHIPPED`].
pub const FIXTURES: &[FamilyFixtures] = &[
    row(qwen2::model::FAMILY, qwen2::model::fixture, &[]),
    row(
        llama::FAMILY,
        llama::fixture,
        &[("mixtral", llama::mixtral_fixture)],
    ),
    row(qwen3::FAMILY, qwen3::fixture, &[]),
    row(olmo2::FAMILY, olmo2::fixture, &[]),
    row(phi3::FAMILY, phi3::fixture, &[]),
    row(smollm3::FAMILY, smollm3::fixture, &[]),
    row(
        granite::FAMILY,
        granite::fixture,
        &[("granitemoe", granite::moe_fixture)],
    ),
    row(mpt::FAMILY, mpt::fixture, &[]),
    row(bloom::FAMILY, bloom::fixture, &[]),
    row(gemma3::FAMILY, gemma3::fixture, &[]),
    row(gemma2::FAMILY, gemma2::fixture, &[]),
    row(qwen3moe::FAMILY, qwen3moe::fixture, &[]),
    row(gpt_oss::FAMILY, gpt_oss::fixture, &[]),
    row(olmoe::FAMILY, olmoe::fixture, &[]),
    row(
        gemma4::FAMILY,
        gemma4::fixture,
        &[("moe", gemma4::moe_fixture)],
    ),
    row(
        minimax_m2::FAMILY,
        minimax_m2::fixture,
        &[("finetune", minimax_m2::finetune_fixture)],
    ),
    row(deepseek2::FAMILY, deepseek2::fixture, &[]),
    row(deepseek3::FAMILY, deepseek3::fixture, &[]),
    row(deepseek32::FAMILY, deepseek32::fixture, &[]),
    row(nemotron_h::FAMILY, nemotron_h::fixture, &[]),
    row(
        deepseek4::FAMILY,
        deepseek4::fixture,
        &[("finetune", deepseek4::finetune_fixture)],
    ),
    row(
        bert::FAMILY,
        bert::fixture,
        &[("cross_encoder", bert::cross_encoder_fixture)],
    ),
    row(
        crate::multimodal::idefics3::FAMILY,
        crate::multimodal::idefics3::fixture,
        &[],
    ),
    row(qwen3next::FAMILY, qwen3next::fixture, &[]),
    row(
        qwen38_27b::FAMILY,
        qwen38_27b::fixture,
        &[
            ("text", qwen38_27b::fixture_text),
            ("fp8", qwen38_27b::fixture_fp8),
            ("finetune", qwen38_27b::finetune_fixture),
        ],
    ),
    row(
        qwen38::FAMILY,
        qwen38::fixture,
        &[
            ("packed", qwen38::packed_fixture),
            ("finetune", qwen38::finetune_fixture),
        ],
    ),
    row(
        glm53_flash::FAMILY,
        glm53_flash::fixture,
        &[("finetune", glm53_flash::finetune_fixture)],
    ),
    row(
        qwen2::model::VL_FAMILY,
        crate::multimodal::qwen_vl::fixture,
        &[],
    ),
];

/// `family`'s checkpoints.
///
/// # Panics
/// When `family` has no row in [`FIXTURES`]: a shipped family needs one.
#[track_caller]
pub fn of(family: FamilyKey) -> &'static FamilyFixtures {
    FIXTURES
        .iter()
        .find(|fixtures| fixtures.family == family)
        .unwrap_or_else(|| panic!("no fixture row for family {family}: add it to FIXTURES"))
}

/// `family`'s base checkpoint.
#[track_caller]
pub fn base(family: FamilyKey) -> Fixture {
    (of(family).base)()
}

/// `family`'s checkpoint for option `name`.
///
/// # Panics
/// When the family has no such option.
#[track_caller]
pub fn variant(family: FamilyKey, name: &str) -> Fixture {
    let fixtures = of(family);
    let (_, make) = fixtures
        .variants
        .iter()
        .find(|(variant, _)| *variant == name)
        .unwrap_or_else(|| panic!("family {family} has no fixture variant {name:?}"));
    make()
}
