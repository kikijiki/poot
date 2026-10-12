//! POOT-725 SC-001 on a device: every [`poot_eval::test_support::i32_gather`] case compiled for the device's target,
//! run through the executor contract and compared word for word with the CPU oracle and the plain-Rust
//! ground truth. One row per backend calls [`check`]; the cases and their comparison live in the fixture.

use poot_eval::test_support::StepFixture;
use poot_eval::test_support::i32_gather::{GatherCase, run_all};
use poot_executor::Executor;
use poot_graph_ir::Storage;
use poot_graph_plan::Target;

use crate::{ConstFixture, run_once};

fn run_case(
    exec: &mut dyn Executor,
    target: Target,
    case: &GatherCase,
) -> Result<poot_tensor::HostTensor, String> {
    let mut consts = Vec::new();
    let mut slots = Vec::new();
    for (id, tensor) in case.inputs(&case.graph) {
        let meta = case.graph.meta(id);
        match meta.storage {
            Storage::Const => {
                let name = meta.name.clone().ok_or("a const without a name")?;
                consts.push(ConstFixture {
                    name: Box::leak(name.into_boxed_str()),
                    tensor,
                });
            }
            Storage::Slot(_) => slots.push(StepFixture {
                key: meta.slot_key().ok_or("a slot without a key")?.clone(),
                tensor,
            }),
            ref other => return Err(format!("unexpected input v{id}: {other:?}")),
        }
    }
    run_once(exec, target, &case.graph, &consts, &slots)
}

/// Run every gather case on `exec` and check it. `bf16` adds the BF16 data cases, for a backend whose BF16
/// table const has a native lane.
pub fn check(backend: &str, bf16: bool, exec: &mut dyn Executor, target: Target) {
    run_all(backend, bf16, |case| run_case(exec, target, case));
}
