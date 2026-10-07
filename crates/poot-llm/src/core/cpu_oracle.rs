//! The CPU-oracle entry the multimodal tests evaluate Runner graphs with (card 545a).
//!
//! A Runner binds its weights as [`Value`]s - dense tensors and packed source components - so the CPU
//! executor is `poot_eval`'s one walk. A packed projection is evaluated as the claimed
//! `PackedContraction` (and a quantized embedding lookup as the claimed `PackedRowGather`). The
//! contraction is a composite (card 629): the walk evaluates its decomposition, which materializes the
//! dequantized `[out, K]` weight and charges it to `PACKED_ORACLE_BUDGET`, so that budget is sized to
//! the largest packed weight the CPU coherence runs evaluate. This is the one place the CPU path
//! claims: the same graph-ir rewrites `compile` runs, exact rewrites of the traced composition.

#[cfg(test)]
use std::collections::HashMap;

#[cfg(test)]
use poot_eval::{EvalBudget, EvalOptions, Value, eval};
#[cfg(test)]
use poot_graph_ir::{Graph, ValueId};
#[cfg(test)]
use poot_graph_plan::{recognize_packed_contractions, recognize_packed_row_gathers};
#[cfg(test)]
use poot_tensor::HostTensor;

#[cfg(test)]
use crate::error::{Result, ResultExt};

/// The ceiling on any one intermediate of a packed contraction's decomposition: 256M elements (1 GiB of
/// F32), the dequantized `[out, K]` weight of the largest real lm_head the CPU coherence runs evaluate
/// (a 1.5B-class vocabulary projection, 151936 x 1536 = 233M elements). A larger weight is a typed
/// `Budget` error naming the contraction, not an allocation.
#[cfg(test)]
const PACKED_ORACLE_BUDGET: EvalBudget = EvalBudget::bounded(256 * 1024 * 1024, 1024 * 1024 * 1024);

/// `g` with its packed linears and packed embedding lookups claimed.
#[cfg(test)]
fn claimed(g: &Graph) -> Graph {
    recognize_packed_row_gathers(&recognize_packed_contractions(g))
}

/// Evaluate a Runner graph with carried state on the CPU oracle; the dense output and the carried
/// state in `g.state` order.
#[cfg(test)]
pub(crate) fn cpu_eval_with_state(
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> Result<(HostTensor, Vec<HostTensor>)> {
    let result = eval(&claimed(g), inputs, EvalOptions::new(PACKED_ORACLE_BUDGET))
        .context("eval with state")?;
    let output = result.output.into_host().context("eval with state")?;
    let state = result
        .state
        .into_iter()
        .map(|value| value.into_host().context("eval with state"))
        .collect::<Result<Vec<_>>>()?;
    Ok((output, state))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use poot_eval::EvalError;
    use poot_graph_ir::{Builder, PackedSourceName, Slot, TensorType, ops};
    use poot_quant::format::{Storage, WeightFormat};
    use poot_quant::{OperandRole, PackedComponentRef, PackedPayload, PackedWeight, SourceRole};

    use super::*;

    /// A Q8_0 payload of `shape` with deterministic pseudo-random blocks whose f16 scale is finite.
    fn q8_0_payload(shape: [usize; 2]) -> Arc<PackedPayload> {
        let Storage::Blocks(layout) = WeightFormat::Q8_0.descriptor().storage else {
            unreachable!("Q8_0 is a block format")
        };
        let descriptor = PackedWeight::try_new(WeightFormat::Q8_0, shape).unwrap();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut bytes: Vec<u8> = (0..descriptor.source_bytes(SourceRole::Blocks))
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect();
        let scale = layout
            .field(OperandRole::Scale)
            .expect("Q8_0 has a scale field");
        let scale_byte = (scale.field.pieces[0].layout.bit(0) / 8) as usize;
        for block in bytes.chunks_mut(layout.bytes) {
            block[scale_byte + 1] &= !0x40;
        }
        Arc::new(
            PackedPayload::try_new(descriptor, [(SourceRole::Blocks, Arc::from(bytes))]).unwrap(),
        )
    }

    /// R-629-1 (card 629 SC-004): a packed projection of a real vocabulary projection's shape (Qwen2.5-0.5B's
    /// lm_head, `[151936, 896]`, synthetic weights) is claimed and evaluated by `cpu_eval` under
    /// `PACKED_ORACLE_BUDGET`, each output the sequential dot product of the activation with the decoded
    /// row. Its dequantized weight is a charged intermediate: under a budget one element short the same
    /// graph is a typed `Budget` error naming that intermediate's element count, not a published value.
    ///
    /// Mutation: charge the intermediate to nothing (drop the one `opts.charge` of
    /// `walk.rs::evaluate_composite`'s loop); the one-element-short budget publishes and the refusal row
    /// goes red.
    #[test]
    fn a_vocabulary_projection_evaluates_under_the_oracle_budget_and_refuses_under_a_smaller_one() {
        let (out, k) = (151_936, 896);
        let owner = q8_0_payload([out, k]);
        let descriptor = owner.weight();
        let b = Builder::new();
        let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![1, k]));
        let logits = ops::packed_linear(&b, x, "lm_head", descriptor, None, None).unwrap();
        let graph = b.finish(logits);

        let activation: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();
        let mut inputs: HashMap<ValueId, Value> = graph
            .inputs
            .iter()
            .filter_map(|&input| {
                let source = graph
                    .meta(input)
                    .name
                    .as_deref()
                    .and_then(PackedSourceName::parse)?;
                Some((
                    input,
                    Value::from(PackedComponentRef::new(Arc::clone(&owner), source.role())),
                ))
            })
            .collect();
        inputs.insert(x.id, HostTensor::f32(vec![1, k], activation.clone()).into());

        let (logits, _) = cpu_eval_with_state(&graph, &inputs)
            .expect("the vocabulary projection fits the budget");
        assert_eq!(logits.shape(), [1, out]);
        let mut row = vec![0.0f32; k];
        for output in [0, 1, 70_000, out - 1] {
            owner.decode_row(output, &mut row).unwrap();
            let mut expected = 0.0f32;
            for (a, w) in activation.iter().zip(&row) {
                expected += a * w;
            }
            assert_eq!(
                logits.as_f32().unwrap()[output].to_bits(),
                expected.to_bits(),
                "logit {output}"
            );
        }

        let one_short = EvalBudget::bounded((out * k - 1) as u64, u64::MAX);
        let refused = eval(&claimed(&graph), &inputs, EvalOptions::new(one_short))
            .expect_err("the dequantized weight is over the budget");
        assert!(
            matches!(
                refused,
                EvalError::Budget {
                    resource: "elements",
                    needed,
                    ..
                } if needed == out * k
            ),
            "{refused:?}"
        );
    }
}
