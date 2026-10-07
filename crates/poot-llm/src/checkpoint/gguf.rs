//! GGUF (qwen2-family) tensor helpers: the permutations and RoPE tables the GGUF readers share.

pub mod permute;
pub mod rope;

pub(crate) use permute::transpose2d;

#[cfg(test)]
mod tests;
