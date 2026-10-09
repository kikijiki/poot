//! Multimodal front ends over the driver: idefics3 / SmolVLM ([`front_end`]) and Qwen2.5-VL
//! ([`qwen_vl`]).

pub mod encoders;
pub mod front_end;
pub mod qwen_vl;
pub(crate) mod qwen_vl_chat;
pub(crate) mod qwen_vl_image;
pub(crate) mod qwen_vl_mrope;
