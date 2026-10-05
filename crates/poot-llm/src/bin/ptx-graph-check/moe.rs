//! Pure graph builders `--precompile` needs: the fixed-dims MoE decode graph (dense/sparse) and the
//! stable-tied-score routing probe cases. No device, no `poot-eval`: this binary only ever compiles
//! these graphs, it never runs them (hardware verification is the pod's job).

use poot_graph_ir::builder::Builder;
use poot_graph_ir::types::TensorType;

// MoE decode (one token) at granite-moe-1b dims (where FFN compute dominates, so the k<<E saving can
// show), so `--precompile` and any future pod MoE run share the kernel cache.
const MOE_E: usize = 32;
const MOE_K: usize = 8;
const MOE_H: usize = 1024;
const MOE_I: usize = 512;

/// Build the MoE decode graph. `sparse` uses the gather-free `IndexedMatMul`/Scatter/Ge path, `false`
/// the dense reference.
pub(crate) fn build_moe(sparse: bool) -> poot_graph_ir::Graph {
    let b = Builder::new();
    let x = b.constant("moe.x", TensorType::f32(vec![1, 1, MOE_H]));
    let rw = b.constant("moe.rw", TensorType::f32(vec![MOE_H, MOE_E]));
    let win = b.constant("moe.win", TensorType::f32(vec![MOE_E, MOE_H, 2 * MOE_I]));
    let wout = b.constant("moe.wout", TensorType::f32(vec![MOE_E, MOE_I, MOE_H]));
    let out = if sparse {
        poot_graph_ir::ops::moe_sparse(&b, x, rw, win, wout, MOE_E, MOE_K, MOE_I)
    } else {
        poot_graph_ir::ops::moe_dense(&b, x, rw, win, wout, MOE_E, MOE_K, MOE_I)
    };
    b.finish(out)
}

#[derive(Clone, Copy)]
pub(crate) struct StableMoeProbeCase {
    pub(crate) k: usize,
    name: &'static str,
}

impl StableMoeProbeCase {
    pub(crate) fn score_const_name(self) -> String {
        format!("stable-ties.scores.{}", self.name)
    }
}

/// Every tied/non-finite/edge-k routing case the stable MoE tie-break coverage exercises: signed zero,
/// canonical non-finites, endpoint underflow, and edge `k` values.
pub(crate) fn stable_moe_probe_cases() -> [StableMoeProbeCase; 8] {
    [
        StableMoeProbeCase {
            name: "all-equal",
            k: 3,
        },
        StableMoeProbeCase {
            name: "nan-low",
            k: 3,
        },
        StableMoeProbeCase {
            name: "positive-infinity",
            k: 2,
        },
        StableMoeProbeCase {
            name: "negative-infinity",
            k: 4,
        },
        StableMoeProbeCase {
            name: "canonical-endpoints",
            k: 4,
        },
        StableMoeProbeCase {
            name: "signed-zero",
            k: 3,
        },
        StableMoeProbeCase {
            name: "k-one",
            k: 1,
        },
        StableMoeProbeCase {
            name: "k-equals-experts",
            k: 5,
        },
    ]
}
