//! Vision-language families: each one's tower, preprocessing, prompt assembly and position machinery,
//! behind the one registry entry that names its checkpoint (ADR-0117 d6). [`front_end`] is the
//! interface the driver crate sees; the families stay private.

pub mod front_end;
pub(crate) mod idefics3;
mod mrope;
pub(crate) mod qwen_vl;

pub(crate) use idefics3::ENTRY as IDEFICS3;
pub(crate) use qwen_vl::ENTRY as QWEN_VL;
