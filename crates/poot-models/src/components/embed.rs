//! The token embedding: rows of the `[vocab, width]` table gathered by token id.

use poot_graph_ir::{Builder, Slot, TensorType, Traced};
use poot_quant::weights::{WeightId, WeightMap, WeightRole};
use poot_tensor::DType;

use super::linear::{Weight, WeightError};
use crate::model::TraceError;

/// The model's `[vocab, width]` embedding table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Embedding {
    table: Weight,
}

impl Embedding {
    /// Resolves the model's embedding table and checks it is `[vocab, width]`.
    pub fn new(map: &WeightMap, vocab: usize, width: usize) -> Result<Self, WeightError> {
        Ok(Self {
            table: Weight::new(map, WeightId::model(WeightRole::Embed), &[vocab, width])?,
        })
    }

    /// The table itself, for a tied head that reuses it as the output projection.
    pub fn table(&self) -> &Weight {
        &self.table
    }

    /// The F32 rows of `tokens` (I32, any shape `[..]`): `[.., width]`.
    pub fn embed(&self, b: &Builder, tokens: Traced) -> Result<Traced, TraceError> {
        let rows = b.gather(self.table.declare(b)?, 0, tokens)?;
        if b.aval(rows)?.dtype == DType::F32 {
            Ok(rows)
        } else {
            Ok(b.cast(rows, DType::F32)?)
        }
    }
}

/// The name of the `Slot::Activation` input holding a step's caller-computed embedding rows,
/// `[rows, tokens, width]` F32.
pub const EMBEDS_SLOT: &str = "embeds";
/// The name of the `Slot::Activation` input marking where those rows apply, `[rows, tokens]` I32:
/// nonzero where the row replaces the token's embedding.
pub const EMBEDS_AT_SLOT: &str = "embeds_at";

/// A step's `StepInput::TokensWithEmbeds` inputs: embedding rows the caller computed (a vision
/// tower's output for a prompt's image placeholders) and the positions they replace.
#[derive(Clone, Copy, Debug)]
pub struct EmbedInputs {
    rows: Traced,
    at: Traced,
}

impl EmbedInputs {
    /// Declare the two inputs for a `[rows, tokens]` step of embedding width `width`.
    pub fn declare(
        b: &Builder,
        rows: usize,
        tokens: usize,
        width: usize,
    ) -> Result<Self, TraceError> {
        Ok(Self {
            rows: b.slot_named(
                Slot::Activation,
                EMBEDS_SLOT,
                TensorType::f32(vec![rows, tokens, width]),
            )?,
            at: b.slot_named(
                Slot::Activation,
                EMBEDS_AT_SLOT,
                TensorType::new(vec![rows, tokens], DType::I32),
            )?,
        })
    }

    /// `x` (`[rows, tokens, width]`, F32) with the caller's rows where [`EMBEDS_AT_SLOT`] is set. A
    /// select, so the unused rows of the input (zeros at the other positions) never reach the result.
    pub fn scatter(&self, b: &Builder, x: Traced) -> Result<Traced, TraceError> {
        let shape = b.aval(x)?.shape;
        let at = b.reshape(self.at, vec![shape[0], shape[1], 1])?;
        Ok(b.select(b.broadcast(at, shape)?, self.rows, x)?)
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
            let b = Builder::for_tests();
            let ids = b
                .slot(Slot::Token, TensorType::new(vec![1, 4], DType::I32))
                .unwrap();
            let y = e.embed(&b, ids).unwrap();
            assert_eq!(b.aval(y).unwrap().shape, vec![1, 4, w]);
            fx.eval(&b.finish(y).unwrap(), &[], &[(Slot::Token, &tokens)], &[])
                .0
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
