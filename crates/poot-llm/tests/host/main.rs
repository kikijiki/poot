//! The model-free host tests of poot-llm as one test binary.
//!
//! Each file used to be its own integration-test binary, and every binary links the whole engine (about
//! 100 MB of executable apiece), so `just test` linked and kept fifteen copies of it. They have no device,
//! no checkpoint and no shared process state, so they live here as modules: the rows keep their names
//! under the module of their old file (`error_sizes::...`).
//!
//! Device, checkpoint and lane-selected tests stay in their own binaries: the lane selections in the
//! justfile and the required-case lists name those binaries.

#[path = "../common/mod.rs"]
mod common;

mod backend_parser;
mod batched_decode;
mod batched_prefill;
mod bf16_checkpoint_plan;
mod capacity;
mod error_sizes;
mod flash_next_budget;
mod gemma3_batched_prefill;
mod gemma4_flash_plan_classify;
mod olmo2_batched_prefill;
mod olmo2_decode;
mod poot_generate_cli;
mod prepared_budgets;
mod qwen35_planned_graph;
mod verify_decode;
