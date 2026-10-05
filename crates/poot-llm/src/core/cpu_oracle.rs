//! The Runner's CPU-oracle entry points (card 545a).
//!
//! A Runner binds its weights as [`Value`]s - dense tensors and packed source components - so the CPU
//! executor it runs is `poot_eval`'s one walk. A packed projection is evaluated as the claimed
//! `PackedContraction` (and a quantized embedding lookup as the claimed `PackedRowGather`), which
//! decode each weight row once; a bare `PackedDequant` would materialize the whole `[out, K]` weight,
//! which `PACKED_ORACLE_BUDGET` caps (the walk moved this from a hardcoded wall inside the
//! evaluator to an opt-in `EvalBudget` the caller states per call, card 554d, so this is now the one
//! place that restates it). This is the one place the CPU path claims (`bind_packed_weights` only
//! places storage; `compile` owns the device claim): the same graph-ir rewrites `compile`
//! runs, and exact rewrites of the traced composition.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use poot_eval::observer::NanTap;
use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{Graph, ValueId};
use poot_graph_plan::{recognize_packed_contractions, recognize_packed_row_gathers};
use poot_tensor::HostTensor;

use crate::error::{Result, ResultExt};

/// The packed-materialization ceiling: 8M elements, 16MB. Carried here (not in `poot-eval`)
/// now that the cap is an `EvalBudget` the caller states, not a hardcoded wall inside the walk.
const PACKED_ORACLE_BUDGET: EvalBudget = EvalBudget::bounded(8 * 1024 * 1024, 16 * 1024 * 1024);

/// Set by [`arm_nan_tap`] (card 554d: `poot_eval::arm_nan_debug_from_env`'s/`DEBUG_NAN`'s successor,
/// `NanTap`, is now a per-call `EvalOptions::observer` rather than a hook `poot-eval` reads itself).
/// When armed, the CPU oracle wires a [`NanTap`] into its own eval calls and logs and disarms on the
/// first non-finite output, same as the deleted hook.
static NAN_TAP_ARMED: AtomicBool = AtomicBool::new(false);

/// Call once at startup to arm the debug hook: a typed choice the binary makes (ADR-0104 decision 5,
/// same rule the retired `--resident` flag followed), never an environment variable read in library
/// code. `main.rs` reads `POOT_DEBUG_NAN` itself and passes the result here; `poot-eval` never reads
/// it (the deleted `poot_eval::arm_nan_debug_from_env` did, this binary opts in explicitly instead).
pub fn arm_nan_tap(enabled: bool) {
    NAN_TAP_ARMED.store(enabled, Ordering::Relaxed);
}

/// Log the tap's first non-finite equation, if any, and disarm (one log line per process, same as the
/// deleted hook's own `DEBUG_NAN.store(false, ..)` after its first hit).
fn log_first_nan(tap: &NanTap) {
    if let Some(hit) = tap.first() {
        tracing::error!(
            eqn = hit.eqn,
            out = hit.out,
            dtype = ?hit.dtype,
            "first non-finite eqn output"
        );
        NAN_TAP_ARMED.store(false, Ordering::Relaxed);
    }
}

/// `g` with its packed linears and packed embedding lookups claimed.
fn claimed(g: &Graph) -> Graph {
    recognize_packed_row_gathers(&recognize_packed_contractions(g))
}

/// Evaluate a stateless Runner graph on the CPU oracle; the dense output.
pub(crate) fn cpu_eval(g: &Graph, inputs: &HashMap<ValueId, Value>) -> Result<HostTensor> {
    let mut tap = NanTap::default();
    let armed = NAN_TAP_ARMED.load(Ordering::Relaxed);
    let opts = EvalOptions::new(PACKED_ORACLE_BUDGET);
    let opts = if armed { opts.observer(&mut tap) } else { opts };
    let result = eval(&claimed(g), inputs, opts).context("eval")?;
    if armed {
        log_first_nan(&tap);
    }
    result.output.into_host().context("eval")
}

/// Evaluate a Runner graph with carried state on the CPU oracle; the dense output and the carried
/// state in `g.state` order.
#[cfg(test)]
pub(crate) fn cpu_eval_with_state(
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> Result<(HostTensor, Vec<HostTensor>)> {
    let mut tap = NanTap::default();
    let armed = NAN_TAP_ARMED.load(Ordering::Relaxed);
    let opts = EvalOptions::new(PACKED_ORACLE_BUDGET);
    let opts = if armed { opts.observer(&mut tap) } else { opts };
    let result = eval(&claimed(g), inputs, opts).context("eval with state")?;
    if armed {
        log_first_nan(&tap);
    }
    let output = result.output.into_host().context("eval with state")?;
    let state = result
        .state
        .into_iter()
        .map(|value| value.into_host().context("eval with state"))
        .collect::<Result<Vec<_>>>()?;
    Ok((output, state))
}
