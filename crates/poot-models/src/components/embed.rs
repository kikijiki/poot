//! The token embedding: rows of the `[vocab, width]` table gathered by token id.

use poot_graph_ir::{Builder, Slot, SlotKey, TensorType, Traced};
use poot_tensor::{DType, HostTensor};
use poot_weights::{WeightId, WeightMap, WeightRole};

use super::linear::{Weight, WeightError};
use crate::io::{
    InputId, InputValue, Layout, PadValue, Per, RequestForm, RequestInput, RequestInputs,
    TokenValues, ValueError,
};
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

/// The request input holding the caller-computed embedding rows a prompt supplies in place of some
/// tokens' own (a multimodal prompt's image rows): sparse per-token values the driver binds on every
/// prefill step that covers one of their positions.
pub const EMBEDS: InputId = InputId::new("embeds");
/// The `Slot::Activation` tag of the embedding rows, `[rows, tokens, width]` F32.
const EMBEDS_TAG: &str = "embeds";
/// The `Slot::Activation` tag marking where those rows apply, `[rows, tokens]` I32: nonzero where the
/// row replaces the token's embedding.
const EMBEDS_AT_TAG: &str = "embeds_at";

/// The schema entry of [`EMBEDS`] for a model of embedding width `width`: optional sparse F32 rows with
/// the presence mask the step scatters them by.
pub fn embeds_input(width: usize) -> RequestInput {
    RequestInput::new(
        EMBEDS,
        SlotKey::new(Slot::Activation, Some(EMBEDS_TAG)),
        RequestForm::new(
            DType::F32,
            Per::Token {
                width,
                layout: Layout::TokenMajor,
                presence: Some(SlotKey::new(Slot::Activation, Some(EMBEDS_AT_TAG))),
            },
            PadValue::F32(0.0),
        ),
    )
}

/// The request inputs of a prompt that supplies embedding rows in place of the embeddings of some of its
/// tokens: `rows` holds `positions.len() * width` values, row-major, one `width`-wide row per entry of
/// `positions` (strictly ascending prompt positions).
pub fn embeds_inputs(
    width: usize,
    positions: Vec<usize>,
    rows: Vec<f32>,
) -> Result<RequestInputs, ValueError> {
    let count = positions.len();
    let values = TokenValues::sparse(positions, HostTensor::f32(vec![count, width], rows))?;
    Ok(RequestInputs::none().with(EMBEDS, InputValue::Tokens(values)))
}

/// A step's [`EMBEDS`] inputs: embedding rows the caller computed (a vision
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
                EMBEDS_TAG,
                TensorType::f32(vec![rows, tokens, width]),
            )?,
            at: b.slot_named(
                Slot::Activation,
                EMBEDS_AT_TAG,
                TensorType::new(vec![rows, tokens], DType::I32),
            )?,
        })
    }

    /// `x` (`[rows, tokens, width]`, F32) with the caller's rows where the presence mask is set. A
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
