//! The output head: hidden states to vocabulary logits through the `[vocab, width]` head weight. A
//! head tied to the embedding is still its own map entry (viewing the embedding's stored tensor),
//! so the head's const has one reader and its storage claim is its own.

use poot_graph_ir::{Builder, Traced};
use poot_quant::weights::{WeightId, WeightMap, WeightRole};

use super::linear::{Weight, WeightError, linear};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Head {
    weight: Weight,
}

impl Head {
    pub fn new(map: &WeightMap, vocab: usize, width: usize) -> Result<Self, WeightError> {
        Ok(Self {
            weight: Weight::new(map, WeightId::model(WeightRole::Head), &[vocab, width])?,
        })
    }

    pub fn weight(&self) -> &Weight {
        &self.weight
    }

    /// `[.., width]` hidden states to `[.., vocab]` logits.
    pub fn logits(&self, b: &Builder, x: Traced) -> Traced {
        linear(b, x, &self.weight, None)
    }
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::TensorType;

    use super::*;
    use crate::components::standard::oracle::{self, Fx};

    fn logits(fx: &Fx, x: &[f32], t: usize, vocab: usize, w: usize) -> Vec<f32> {
        let head = Head::new(&fx.map(), vocab, w).unwrap();
        let b = Builder::new();
        let xs = b.constant("x", TensorType::f32(vec![t, w]));
        let y = head.logits(&b, xs);
        fx.eval(&b.finish(y), &[("x", x)], &[], &[]).0
    }

    fn reference(x: &[f32], table: &[f32], vocab: usize, w: usize) -> Vec<f64> {
        x.chunks(w)
            .flat_map(|row| {
                (0..vocab).map(move |v| {
                    (0..w)
                        .map(|j| f64::from(row[j]) * f64::from(table[v * w + j]))
                        .sum::<f64>()
                })
            })
            .collect()
    }

    /// SC-001: the head reads its own weight against an f64 `x H^T`, and only that weight matters;
    /// a tied head (its map entry viewing the embedding's tensor) is its own `w.head` const with the
    /// embedding's values.
    #[test]
    fn head_reads_its_own_entry_tied_or_not() {
        let (t, vocab, w) = (2, 5, 4);
        let (embed, head) = (
            WeightId::model(WeightRole::Embed),
            WeightId::model(WeightRole::Head),
        );
        let x = oracle::random(t * w, 1);

        let mut untied = Fx::new(4);
        untied.f32(embed, &[vocab, w]);
        untied.bf16(head, &[vocab, w]);
        let got = logits(&untied, &x, t, vocab, w);
        oracle::assert_matches_f64(&got, &reference(&x, &untied.values(head), vocab, w), 1e-5);
        let mut p = untied.clone();
        p.perturb(head);
        oracle::assert_differs(&logits(&p, &x, t, vocab, w), &got, head);
        let mut p = untied.clone();
        p.perturb(embed);
        assert_eq!(
            logits(&p, &x, t, vocab, w),
            got,
            "an untied head ignores the embedding"
        );

        let mut tied = Fx::new(5);
        tied.f32(embed, &[vocab, w]);
        tied.tie(head, embed);
        let h = Head::new(&tied.map(), vocab, w).unwrap();
        assert_eq!(h.weight().id(), head);
        let got = logits(&tied, &x, t, vocab, w);
        oracle::assert_matches_f64(&got, &reference(&x, &tied.values(embed), vocab, w), 1e-5);
        let mut p = tied.clone();
        p.perturb(embed);
        oracle::assert_differs(&logits(&p, &x, t, vocab, w), &got, embed);
    }
}
