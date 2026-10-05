//! Runner state, graph binding, and generation policy shared by model architectures and backends.

pub(crate) mod batched;
pub(crate) mod cpu_oracle;
pub(crate) mod decode_arch;
pub(crate) mod decode_step;
pub(crate) mod generate;
pub(crate) mod graphs;
pub(crate) mod modality;
pub(crate) mod runner;
pub(crate) mod sampler;
pub(crate) mod speculative;
