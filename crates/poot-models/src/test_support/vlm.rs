//! The in-memory fixtures of the shipped vision-language families, for the serving tests that run a
//! front end on a device: the driver crate's tests take them from here instead of looking a family up by
//! its name.

use crate::registry::Fixture;

/// The SmolVLM (idefics3) fixture: 8-pixel frames and a llama text tower.
#[must_use]
pub fn smolvlm_fixture() -> Fixture {
    (crate::multimodal::IDEFICS3.fixture)()
}

/// The Qwen2.5-VL fixture: 2-pixel patches that merge 2 x 2 and the qwen2 text tower.
#[must_use]
pub fn qwen_vl_fixture() -> Fixture {
    (crate::multimodal::QWEN_VL.fixture)()
}
