//! POOT-1195 (R2-C1-001): the oracle's `SampleToken(GumbelTopKTopP)` keeps the real top-p nucleus.
//!
//! The nucleus is observed through the production entry point (`poot_eval::eval` of a `SampleToken`
//! graph), never by calling the mass helper: a row of noise that is zero everywhere but one huge spike
//! makes the Gumbel argmax return the spiked token exactly when that token survived the truncation, and
//! the top token otherwise. The nucleus is a prefix of the descending-logit order, so a binary search
//! over the spiked index finds its size. The exact reference below lives only in this file (sorted
//! descending, f64 cumulative sum); no production code imports it.

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::op::SampleRule;
use poot_graph_ir::{Builder, TensorType, ValueId};
use poot_tensor::{DType, HostTensor};

/// The smallest prefix of the probability-sorted tokens whose f64 mass reaches `top_p` of the total.
/// `logits` are already strictly descending, so the sorted order is the index order.
fn exact_nucleus_size(logits: &[f32], temperature: f32, top_p: f32) -> usize {
    let max = f64::from(logits[0]);
    let mass: Vec<f64> = logits
        .iter()
        .map(|&l| ((f64::from(l) - max) / f64::from(temperature)).exp())
        .collect();
    let total: f64 = mass.iter().sum();
    let target = f64::from(top_p) * total;
    let mut cum = 0.0f64;
    for (i, m) in mass.iter().enumerate() {
        cum += m;
        if cum >= target {
            return i + 1;
        }
    }
    logits.len()
}

/// `p_i proportional to (i+1)^-s`, as descending logits.
fn zipf_logits(vocab: usize, s: f64) -> Vec<f32> {
    (0..vocab)
        .map(|i| (-s * ((i + 1) as f64).ln()) as f32)
        .collect()
}

/// A top token of probability 0.5 and 500 tail tokens of about 0.001 each. The tail is made strictly
/// descending (a 1e-4 relative step per token) so the nucleus boundary is a single token, not a tie
/// group the value-based cut keeps or drops as a whole.
fn head_and_flat_tail_logits() -> Vec<f32> {
    let mut logits = vec![0.5f64.ln() as f32];
    logits.extend((0..500).map(|j| (0.001f64 * (1.0 - 1.0e-4 * j as f64)).ln() as f32));
    logits
}

struct Distribution {
    name: &'static str,
    logits: Vec<f32>,
    top_p: f32,
    /// The exact reference's nucleus size, pinned by the card for the two Zipf rows.
    pinned: Option<usize>,
}

fn distributions() -> Vec<Distribution> {
    vec![
        Distribution {
            name: "zipf s=1.1 V=32000 top_p=0.95",
            logits: zipf_logits(32_000, 1.1),
            top_p: 0.95,
            pinned: Some(12_413),
        },
        Distribution {
            name: "zipf s=0.8 V=32000 top_p=0.95",
            logits: zipf_logits(32_000, 0.8),
            top_p: 0.95,
            pinned: Some(25_496),
        },
        Distribution {
            name: "top 0.5 + 500 x 0.001 top_p=0.9",
            logits: head_and_flat_tail_logits(),
            top_p: 0.9,
            pinned: None,
        },
    ]
}

/// One `SampleToken(GumbelTopKTopP)` graph over `[1, vocab]` rows.
struct TopPGraph {
    graph: poot_graph_ir::Graph,
    logits: ValueId,
    noise: ValueId,
    params: ValueId,
    top_k: ValueId,
    vocab: usize,
}

impl TopPGraph {
    fn new(vocab: usize) -> Self {
        let b = Builder::for_tests();
        let logits = b
            .constant("logits", TensorType::f32(vec![1, vocab]))
            .unwrap();
        let noise = b
            .constant("noise", TensorType::f32(vec![1, vocab]))
            .unwrap();
        let params = b.constant("params", TensorType::f32(vec![1, 4])).unwrap();
        let top_k = b
            .constant("top_k", TensorType::new(vec![1], DType::I32))
            .unwrap();
        let out = b
            .sample_token(
                SampleRule::GumbelTopKTopP,
                logits,
                Some(noise),
                Some(params),
                Some(top_k),
            )
            .unwrap();
        let graph = b.finish(out).unwrap();
        Self {
            graph,
            logits: logits.id(),
            noise: noise.id(),
            params: params.id(),
            top_k: top_k.id(),
            vocab,
        }
    }

    /// The token drawn for `logits` with `noise` at `(temperature, top_p)`, top-k disabled, min-p off.
    fn token(&self, logits: &[f32], noise: Vec<f32>, temperature: f32, top_p: f32) -> usize {
        assert_eq!(logits.len(), self.vocab);
        let mut inputs: HashMap<ValueId, Value> = HashMap::new();
        inputs.insert(
            self.logits,
            HostTensor::f32(vec![1, self.vocab], logits.to_vec()).into(),
        );
        inputs.insert(
            self.noise,
            HostTensor::f32(vec![1, self.vocab], noise).into(),
        );
        inputs.insert(
            self.params,
            HostTensor::f32(
                vec![1, 4],
                vec![1.0 / temperature, f32::NEG_INFINITY, 1.0, top_p],
            )
            .into(),
        );
        inputs.insert(self.top_k, HostTensor::i32(vec![1], vec![0]).into());
        let out = eval(
            &self.graph,
            &inputs,
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap();
        let tensor = out.output.into_host().unwrap();
        let words = tensor.as_i32().unwrap();
        assert_eq!(words[1], -1, "no non-finite logit in these rows");
        words[0] as usize
    }

    /// Whether the oracle keeps token `i`: a spike of noise at `i` wins exactly when `i` survived.
    fn keeps(&self, logits: &[f32], i: usize, top_p: f32) -> bool {
        let mut noise = vec![0.0f32; self.vocab];
        noise[i] = 1.0e6;
        self.token(logits, noise, 1.0, top_p) == i
    }

    /// The nucleus size: the largest kept index plus one (the nucleus is a prefix of descending logits).
    fn nucleus_size(&self, logits: &[f32], top_p: f32) -> usize {
        let (mut lo, mut hi) = (0usize, self.vocab - 1);
        assert!(self.keeps(logits, 0, top_p), "the top token is always kept");
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if self.keeps(logits, mid, top_p) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo + 1
    }
}

/// SC-002: the oracle's nucleus equals the exact f64 reference on the three distributions.
#[test]
fn oracle_nucleus_equals_the_exact_reference() {
    for d in distributions() {
        let reference = exact_nucleus_size(&d.logits, 1.0, d.top_p);
        if let Some(pinned) = d.pinned {
            assert_eq!(reference, pinned, "{}: the card pins the reference", d.name);
        }
        let graph = TopPGraph::new(d.logits.len());
        let got = graph.nucleus_size(&d.logits, d.top_p);
        assert_eq!(got, reference, "{}: oracle nucleus vs exact f64", d.name);
    }
}
