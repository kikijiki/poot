//! Model-neutral graph compositions shared by multiple architecture families.
//!
//! A component belongs here only when its graph semantics are genuinely shared. Architecture-specific
//! scheduling, source naming, and policy stay with the architecture even when their implementations look
//! similar.

pub mod attention;
pub mod dims;
pub mod embed;
pub mod ffn;
pub mod head;
mod kv_pool;
pub mod linear;
pub mod moe_decode;
pub mod moe_prefill;
pub mod norm;
pub mod rope;
pub mod standard;
#[cfg(test)]
pub(crate) mod testing;

pub use kv_pool::*;
