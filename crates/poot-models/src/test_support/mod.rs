//! Fixtures built through the shipped tracers, for the tests, benches and examples of every crate above the
//! tracers (ADR-0113: a fixture lives in the crate that owns what it exercises; `poot-executor-parity` sits
//! below this crate and holds only the model-free parity machinery).

#[cfg(feature = "test-support")]
pub mod dense;
#[cfg(feature = "test-support")]
pub mod family_fixtures;
#[cfg(feature = "test-support")]
pub mod fixtures;
#[cfg(feature = "test-support")]
pub mod pipeline;
#[cfg(feature = "test-support")]
pub mod real_dims;
#[cfg(feature = "test-support")]
pub mod vlm;
#[cfg(feature = "test-support")]
pub mod weight_map;
