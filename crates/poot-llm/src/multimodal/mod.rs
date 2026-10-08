//! Multimodal front ends: the idefics3 / SmolVLM front end on the driver, and Qwen2.5-VL's request
//! processing (held for POOT-650, which puts its vision tower and mRoPE front end on the driver).

pub mod encoders;
pub mod front_end;
pub(crate) mod mrope;
pub(crate) mod qwen_vl_chat;
pub(crate) mod qwen_vl_image;
pub(crate) mod qwen_vl_mrope;
