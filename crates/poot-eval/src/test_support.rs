//! CPU-oracle fixtures with no production caller, for the executor, planner and device tests above this crate
//! (card 643, card 671: dead under the strict dead-pub rule, a test reference is never a use): the I32 gather
//! graphs and their oracle, the live-attention graph, and the oracle for a graph whose weights bind through a
//! `WeightMap`.

pub mod i32_gather;
pub mod i32_slot_gather;
pub mod i32_table_gather;
pub mod live_attention;
pub mod stage_chain;
pub mod weight_map_oracle;

/// One step's slot value for a fixture graph: the slot's key and its typed host tensor. Defined beside the
/// shared CPU oracle ([`weight_map_oracle::oracle_for`]) that takes it, so `poot-executor-parity`, `poot_models::test_support`
/// and the device crates name one type.
pub struct StepFixture {
    pub key: poot_graph_ir::SlotKey,
    pub tensor: poot_tensor::HostTensor,
}
