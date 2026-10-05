//! The token embedding: rows of the `[vocab, width]` table gathered by token id.

use poot_graph_ir::{Builder, Traced};
use poot_quant::weights::{WeightId, WeightMap, WeightRole};
use poot_tensor::DType;

use super::linear::{Weight, WeightError};

/// The model's `[vocab, width]` embedding table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Embedding {
    table: Weight,
}

impl Embedding {
    pub fn new(map: &WeightMap, vocab: usize, width: usize) -> Result<Self, WeightError> {
        Ok(Self {
            table: Weight::new(map, WeightId::model(WeightRole::Embed), &[vocab, width])?,
        })
    }

    pub fn table(&self) -> &Weight {
        &self.table
    }

    /// The F32 rows of `tokens` (I32, any shape `[..]`): `[.., width]`.
    pub fn embed(&self, b: &Builder, tokens: Traced) -> Traced {
        let rows = b.gather(self.table.declare(b), 0, tokens);
        if b.aval(rows).dtype == DType::F32 {
            rows
        } else {
            b.cast(rows, DType::F32)
        }
    }
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{Slot, TensorType};

    use super::*;
    use crate::components::standard::oracle::{self, Fx};

    /// SC-001: the embedding of a `[1, 4]` token block (a repeated id included) is the table's
    /// rows, read from BF16 storage; the table matters.
    #[test]
    fn embed_gathers_the_table_rows_of_each_token() {
        let (vocab, w) = (6, 5);
        let id = WeightId::model(WeightRole::Embed);
        let mut fx = Fx::new(8);
        fx.bf16(id, &[vocab, w]);
        let tokens = [5, 2, 2, 0];
        let run = |fx: &Fx| {
            let e = Embedding::new(&fx.map(), vocab, w).unwrap();
            let b = Builder::new();
            let ids = b.slot(Slot::Token, TensorType::new(vec![1, 4], DType::I32));
            let y = e.embed(&b, ids);
            assert_eq!(b.aval(y).shape, vec![1, 4, w]);
            fx.eval(&b.finish(y), &[], &[(Slot::Token, &tokens)], &[]).0
        };
        let got = run(&fx);
        let table = fx.values(id);
        let want: Vec<f64> = tokens
            .iter()
            .flat_map(|&t| table[t as usize * w..][..w].iter().map(|&v| f64::from(v)))
            .collect();
        oracle::assert_matches_f64(&got, &want, 0.0);
        let mut p = fx.clone();
        p.perturb(id);
        oracle::assert_differs(&run(&p), &got, id);
    }
}
