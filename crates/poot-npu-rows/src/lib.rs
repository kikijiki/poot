//! Device rows that run a real model's step on the Intel NPU.
//!
//! A row here needs the model crate (it traces the model's step) and the NPU device (it runs the compiled
//! regions), and the dependency ranks allow no lower crate both: `poot-npu` sits below `poot-models`. This crate
//! sits above both, holds no library code, and every test of it dispatches on the NPU, so the model-free lane
//! excludes it and `just test-device-vpu` runs it.
