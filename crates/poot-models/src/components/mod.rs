//! Model-neutral graph compositions shared by multiple architecture families.
//!
//! A component belongs here only when its graph semantics are genuinely shared. Architecture-specific
//! scheduling, source naming, and policy stay with the architecture even when their implementations look
//! similar.

pub mod attention;
pub mod conv;
pub mod dims;
pub mod dsa;
pub mod embed;
pub mod ffn;
pub mod gdn;
pub(crate) mod gdn_names;
pub mod head;
pub mod kda;
pub mod linear;
pub mod mamba2;
pub mod mhc;
pub mod mla;
pub mod moe;
pub mod norm;
pub mod ple;
pub mod rope;
pub mod router;
pub(crate) mod sparse_top_k;
pub mod standard;
pub mod standard_family;
#[cfg(test)]
pub(crate) mod testing;
pub mod witness;
