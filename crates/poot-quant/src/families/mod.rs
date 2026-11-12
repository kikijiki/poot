//! The scalar semantics of each value family ([`crate::format::ValueMap`]): what a code means.
//!
//! A format's descriptor says where its bits are and which family reads them; the family says what
//! the bits are worth. Each family is written once here and read by both the reference interpreter
//! ([`crate::decode`]) and the const plan ([`crate::plan`]), so the two cannot disagree about a
//! value. The device emitters in `poot-kernelgen` have one module per family as well
//! (`packed/family/`), selected by the same enum.

pub mod float;
pub mod integer;
pub mod lattice;
