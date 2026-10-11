//! DeepSeek-V4's routed feed-forward. Every layer routes through [`components::moe`] with the
//! square-root-softplus scoring; the first `num_hash_layers` layers do not rank the experts but read
//! them from a `[vocab, top_k]` table indexed by the token id. A hash layer scores the experts
//! exactly like a score-routed one and gates the table's experts with those scores (normalized over
//! the selection, then scaled): only where the ids come from differs, so the hash router is this
//! family's own and hands its routes to the shared experts.
//!
//! [`components::moe`]: crate::components::moe

use poot_graph_ir::{BinOp, Builder, IndexRange, RedOp, Scalar, TensorType, Traced, ops};
use poot_quant::weights::{HandleFormat, MoeRole, WeightId, WeightMap, WeightRole};
use poot_tensor::DType;

use super::config::DeepseekV4Config;
use crate::components::linear::{WeightError, linear};
use crate::components::moe::{self, MoeWeights};
use crate::components::router::{RouterKind, Routes, scores};
use crate::components::standard::Step;
use crate::components::witness::{self, WitnessKind};

/// A hash-routed layer's table of the experts each token id selects.
#[derive(Debug)]
pub(super) struct HashTable {
    id: WeightId,
    vocab: usize,
    top_k: usize,
    /// `[0, experts)`: every word of the table is an expert id, checked when the weight binds.
    experts: IndexRange,
}

impl HashTable {
    pub fn new(
        map: &WeightMap,
        layer: usize,
        vocab: usize,
        top_k: usize,
        experts: usize,
    ) -> Result<Self, WeightError> {
        let id = WeightId::layer(layer, WeightRole::Moe(MoeRole::HashTable));
        let handle = map.handle(id).ok_or(WeightError::Unmapped { id })?;
        if handle.shape != [vocab, top_k] {
            return Err(WeightError::Shape {
                id,
                expected: vec![vocab, top_k],
                found: handle.shape.clone(),
            });
        }
        if handle.format != HandleFormat::Dense(DType::I32) {
            return Err(WeightError::Format {
                id,
                format: handle.format,
            });
        }
        let experts = IndexRange::below(experts).ok_or(WeightError::IndexBound {
            id,
            field: "n_routed_experts",
            bound: experts,
        })?;
        Ok(Self {
            id,
            vocab,
            top_k,
            experts,
        })
    }

    /// The expert ids of each token of the step, `[rows * tokens, top_k]`, as the I32 words the
    /// router's selector reads: the tokens gather rows of the I32 table exactly.
    fn ids(&self, b: &Builder, tokens: Traced) -> Traced {
        let table = b.constant_ranged(
            &self.id.const_name(),
            TensorType::new(vec![self.vocab, self.top_k], DType::I32),
            self.experts,
        );
        let flat = b.reshape(tokens, vec![b.aval(tokens).numel()]);
        b.gather(table, 0, flat)
    }
}

/// A layer's routed feed-forward weights: the experts and router every layer has, and for a
/// hash-routed layer the table its experts come from.
#[derive(Debug)]
pub(super) struct RoutedWeights {
    moe: MoeWeights,
    table: Option<HashTable>,
}

impl RoutedWeights {
    pub fn new(map: &WeightMap, layer: usize, cfg: &DeepseekV4Config) -> Result<Self, WeightError> {
        let hash = cfg.is_hash_routed(layer);
        Ok(Self {
            moe: MoeWeights::new(
                map,
                layer,
                if hash { &cfg.hash_moe } else { &cfg.score_moe },
            )?,
            table: hash
                .then(|| {
                    HashTable::new(
                        map,
                        layer,
                        cfg.vocab,
                        cfg.hash_moe.top_k(),
                        cfg.hash_moe.experts(),
                    )
                })
                .transpose()?,
        })
    }

    /// The routed block over `x` (`[rows, tokens, width]`) of layer `layer`. `tokens` are the
    /// step's token ids bounded by the vocabulary ([`super::stack::Tower::trace`]), which a hash
    /// layer indexes its table with.
    pub fn trace(
        &self,
        b: &Builder,
        step: &Step,
        layer: usize,
        x: Traced,
        tokens: Traced,
        cfg: &DeepseekV4Config,
    ) -> Traced {
        match &self.table {
            None => moe::moe(b, step, layer, x, None, &self.moe, &cfg.score_moe),
            Some(table) => {
                let routes = hash_routes(b, step, layer, x, tokens, &self.moe, table, cfg);
                moe::experts(b, x, routes, &self.moe, &cfg.hash_moe)
            }
        }
    }
}

/// The routes of a hash layer: the table's experts for each token, weighted by their scores
/// normalized over the selection and scaled. The three run-time witnesses are the score-routed
/// router's: finite logits, an in-range selection and a usable selected sum.
#[allow(clippy::too_many_arguments)]
fn hash_routes(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    tokens: Traced,
    weights: &MoeWeights,
    table: &HashTable,
    cfg: &DeepseekV4Config,
) -> Routes {
    let experts = cfg.hash_moe.experts();
    let k = table.top_k;
    let width = cfg.width;
    let shape = step.shape();
    let count = shape.rows.get() * shape.tokens.get();

    let x = b.reshape(x, vec![count, width]);
    let logits = linear(b, x, weights.router().weight(), None);
    witness::declare(
        step,
        layer,
        WitnessKind::RouterLogitsFinite,
        witness::nonfinite_count(b, logits),
    );
    let ids = table.ids(b, tokens);
    witness::declare(
        step,
        layer,
        WitnessKind::SelectorInRange,
        witness::out_of_range_count(b, ids, experts),
    );

    // The score of each selected expert: a one-hot over the expert axis, an exact I32 `0`/`1` that
    // widens to F32. Canonical scores keep an overflowing one finite, so an unselected expert's
    // `0 * score` is never NaN.
    let scores =
        ops::canonical_router_scores(b, scores(b, logits, RouterKind::SqrtSoftplusBiasedTopK));
    let experts_row = b.broadcast(
        b.reshape(b.iota_i32(experts), vec![1, 1, experts]),
        vec![count, k, experts],
    );
    let ids_grid = b.broadcast(b.reshape(ids, vec![count, k, 1]), vec![count, k, experts]);
    let one_hot = b.cast(
        b.binary(
            BinOp::Mul,
            b.binary(BinOp::Ge, ids_grid, experts_row),
            b.binary(BinOp::Ge, experts_row, ids_grid),
        ),
        DType::F32,
    );
    let scores = b.broadcast(
        b.reshape(scores, vec![count, 1, experts]),
        vec![count, k, experts],
    );
    let selected = b.reduce(RedOp::Sum, b.binary(BinOp::Mul, one_hot, scores), 2, false);
    let sum = b.reduce(RedOp::Sum, selected, 1, true);
    witness::declare(
        step,
        layer,
        WitnessKind::SelectedSumUsable,
        witness::unusable_sum_count(b, sum),
    );
    Routes {
        ids,
        weights: b.binary_scalar(
            BinOp::Mul,
            b.binary(BinOp::Div, selected, sum),
            Scalar::F32(cfg.route_scale),
        ),
    }
}
