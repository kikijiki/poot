use poot_tensor::DType;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use poot_graph_ir::{Builder, Graph, OpKind, ShapeError, Slot, TensorType, ValueId};
use poot_load::packed_safetensors::{
    AuthenticatedSafetensorsHandleSet, ExactSourceOwner, ExactSourceOwnerCache, InventoryDecision,
    PackedArtifactManifest, PackedOwnerCache, PackedSafetensorsLimits, PackedShardManifest,
    TensorDisposition, sha256_digest,
};

use super::helpers;
use crate::exact_dense::{DenseOwnerTensorView, ExactDenseError};
use crate::{EvalBudget, EvalError, EvalOptions, ExactBf16Error, ExactValue, Value, eval};
use poot_tensor::HostTensor;

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

/// Card 534a: `fold_dense_contractions`/`fold_dense_bf16_row_gathers` moved
/// into `poot-graph-plan`, `pub(crate)` there, reachable only through `prepare_target_graph`. This
/// crate (the CPU evaluator) must never depend on `poot-graph-plan`, even for tests -
/// `poot_eval_does_not_depend_on_graph_plan` (`exact_i32.rs`) guards the Cargo.toml directly, since
/// the oracle stays below the planner. `Builder` has no method for either op either way (compiler-only:
/// `op.rs`'s doc), so each fixture below traces the pre-fold chain normally, then hand-rewrites it to
/// the exact post-fold shape the real pass would produce for that one chain - not a
/// re-implementation of the pass's general pattern match, just this fixture's own known shape,
/// preserving every value id the eval-side assertions key on (`logits.id`, `widened.id`).
fn as_dense_contraction(mut g: Graph, x: ValueId, weight: ValueId) -> Graph {
    let dtype = g.aval(weight).dtype;
    let out = g.output;
    g.eqns = vec![poot_graph_ir::Eqn {
        op: OpKind::DenseContraction { weight: dtype },
        inputs: vec![
            poot_graph_ir::Operand::Value(x),
            poot_graph_ir::Operand::Value(weight),
        ],
        out,
        layer: None,
    }];
    g
}

/// See [`as_dense_contraction`]'s doc. Only valid when the traced chain is a plain
/// `Cast(F32, Gather(table, index))` with no `Reshape` link (`dense_row_gather_matches_gather_then_widen`'s
/// reshaped case hand-rewrites its own two-equation shape inline instead).
fn as_dense_row_gather(mut g: Graph, table: ValueId, index: ValueId) -> Graph {
    let dtype = g.aval(table).dtype;
    let out = g.output;
    g.eqns = vec![poot_graph_ir::Eqn {
        op: OpKind::DenseRowGather { source: dtype },
        inputs: vec![
            poot_graph_ir::Operand::Value(table),
            poot_graph_ir::Operand::Value(index),
        ],
        out,
        layer: None,
    }];
    g
}

/// BF16 words with their f32 values spelled as literals, so a decode with another exponent bias, byte order,
/// or mantissa placement cannot reproduce them.
const WORDS: [(u16, f32); 8] = [
    (0x3f80, 1.0),
    (0xc000, -2.0),
    (0x3f00, 0.5),
    (0x4040, 3.0),
    (0xbf40, -0.75),
    (0x3fc0, 1.5),
    (0x4100, 8.0),
    (0xbe80, -0.25),
];

struct Fixture {
    path: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Load authenticated owners for `(name, dtype, shape, bytes)` rows.
fn owners(rows: &[(&str, &str, &[usize], Vec<u8>)]) -> HashMap<String, Arc<ExactSourceOwner>> {
    let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
    let fixture = Fixture {
        path: std::env::temp_dir()
            .join(format!("poot-exact-bf16-{}-{sequence}", std::process::id())),
    };
    fs::create_dir(&fixture.path).unwrap();
    let config = b"{}";
    fs::write(fixture.path.join("config.json"), config).unwrap();
    let mut header = String::from("{");
    let mut data = Vec::new();
    for (ordinal, (name, dtype, shape, bytes)) in rows.iter().enumerate() {
        let start = data.len();
        data.extend_from_slice(bytes);
        if ordinal > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            r#""{name}":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[{start},{}]}}"#,
            data.len()
        ));
    }
    header.push('}');
    let header = header.into_bytes();
    let mut shard = (header.len() as u64).to_le_bytes().to_vec();
    shard.extend_from_slice(&header);
    shard.extend_from_slice(&data);
    fs::write(fixture.path.join("model.safetensors"), &shard).unwrap();
    let index = format!(
        r#"{{"weight_map":{{{}}}}}"#,
        rows.iter()
            .map(|(name, ..)| format!(r#""{name}":"model.safetensors""#))
            .collect::<Vec<_>>()
            .join(",")
    )
    .into_bytes();
    fs::write(fixture.path.join("model.safetensors.index.json"), &index).unwrap();
    let manifest = PackedArtifactManifest {
        repository: "local/exact-bf16-fixture".to_string(),
        revision: "card376".to_string(),
        config_length: config.len(),
        config_sha256: sha256_digest(config),
        index_sha256: sha256_digest(&index),
        shards: vec![PackedShardManifest {
            filename: "model.safetensors".to_string(),
            file_length: shard.len(),
            file_sha256: sha256_digest(&shard),
            header_length: header.len(),
            header_sha256: sha256_digest(&header),
        }],
    };
    let limits = PackedSafetensorsLimits {
        config_bytes: 1024,
        index_bytes: 4096,
        header_bytes_per_shard: 4096,
        shard_count: 1,
        tensor_entries: rows.len(),
        selected_source_bytes: 4096,
        packed_source_bytes: 1,
    };
    let mut authenticated =
        AuthenticatedSafetensorsHandleSet::authenticate(&fixture.path, manifest, limits).unwrap();
    let result = authenticated
        .load_mixed(
            &mut PackedOwnerCache::new(),
            &mut ExactSourceOwnerCache::new(),
            |inventory| {
                inventory
                    .rows()
                    .map(|row| {
                        let disposition = match row.dtype() {
                            "BF16" => TensorDisposition::DenseBf16,
                            "F32" => TensorDisposition::DenseF32,
                            _ => return Err("unexpected fixture dtype"),
                        };
                        Ok(InventoryDecision::new(row.key(), disposition))
                    })
                    .collect::<Result<Vec<_>, _>>()
            },
        )
        .unwrap();
    result
        .exact_metadata
        .into_iter()
        .map(|(name, meta)| (name, meta.owner))
        .collect()
}

fn bf16_bytes(words: impl IntoIterator<Item = u16>) -> Vec<u8> {
    words.into_iter().flat_map(u16::to_le_bytes).collect()
}

/// One BF16 constant named `table` holding `WORDS[i]` at element `i`.
fn table(shape: &[usize]) -> DenseOwnerTensorView {
    let numel = shape.iter().product::<usize>();
    let bytes = bf16_bytes(WORDS[..numel].iter().map(|&(word, _)| word));
    let owners = owners(&[("table", "BF16", shape, bytes)]);
    DenseOwnerTensorView::new(Arc::clone(&owners["table"])).unwrap()
}

fn dense_bits(value: &Value) -> Vec<u32> {
    let Value::Host(tensor) = value else {
        panic!("expected a dense output, got {value:?}");
    };
    tensor
        .to_f32()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

#[test]
fn bf16_cast_decodes_owner_words_bit_exactly() {
    let words = [0x3f80, 0xc000, 0x0000, 0x8000, 0x7fc1, 0xff80];
    let owners = owners(&[("source", "BF16", &[2, 3], bf16_bytes(words))]);
    let view = DenseOwnerTensorView::new(Arc::clone(&owners["source"])).unwrap();
    let builder = Builder::new();
    let source = builder.constant("source", TensorType::new([2, 3], DType::BF16));
    let widened = builder.cast(source, DType::F32);
    let graph = builder.finish(widened);

    let output = eval(
        &graph,
        &HashMap::from([(source.id, Value::from(view))]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|r| r.output);
    let expected = [
        1.0f32.to_bits(),
        (-2.0f32).to_bits(),
        0.0f32.to_bits(),
        (-0.0f32).to_bits(),
        0x7fc1_0000,
        f32::NEG_INFINITY.to_bits(),
    ];
    assert_eq!(dense_bits(&output.unwrap()), expected);
}

#[test]
fn bf16_gather_reads_the_selected_rows() {
    let builder = Builder::new();
    let source = builder.constant("table", TensorType::new([4, 2], DType::BF16));
    let token = builder.slot(Slot::Token, TensorType::new([3], DType::I32));
    let rows = builder.gather(source, 0, token);
    let widened = builder.cast(rows, DType::F32);
    let graph = builder.finish(widened);
    let inputs = |ids: Vec<i32>| {
        HashMap::from([
            (source.id, Value::from(table(&[4, 2]))),
            (token.id, Value::Host(HostTensor::i32(vec![3], ids))),
        ])
    };

    let output = eval(
        &graph,
        &inputs(vec![2, 0, 3]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|r| r.output);
    let expected = [
        WORDS[4].1, WORDS[5].1, WORDS[0].1, WORDS[1].1, WORDS[6].1, WORDS[7].1,
    ];
    assert_eq!(dense_bits(&output.unwrap()), expected.map(f32::to_bits));

    for (ids, position, index, kind) in [
        (
            vec![1, 4, 0],
            1,
            4,
            crate::ops::index_rule::IndexFaultKind::OutOfRange,
        ),
        (
            vec![-1, 0, 0],
            0,
            -1,
            crate::ops::index_rule::IndexFaultKind::Negative,
        ),
    ] {
        let output = eval(
            &graph,
            &inputs(ids),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output);
        assert!(matches!(
            &output,
            Err(EvalError::Index(fault))
                if fault.position == position
                    && fault.value == crate::IndexValue::I32(index)
                    && fault.len == 4
                    && fault.kind == kind
        ));
    }

    // A non-leading axis selects columns.
    let builder = Builder::new();
    let source = builder.constant("table", TensorType::new([2, 3], DType::BF16));
    let token = builder.slot(Slot::Token, TensorType::new([2], DType::I32));
    let columns = builder.cast(builder.gather(source, 1, token), DType::F32);
    let graph = builder.finish(columns);
    let output = eval(
        &graph,
        &HashMap::from([
            (source.id, Value::from(table(&[2, 3]))),
            (token.id, Value::Host(HostTensor::i32(vec![2], vec![2, 0]))),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|r| r.output);
    let expected = [WORDS[2].1, WORDS[0].1, WORDS[5].1, WORDS[3].1];
    assert_eq!(dense_bits(&output.unwrap()), expected.map(f32::to_bits));
}

/// The Qwen3.8 LM head: an F32 activation times a transposed BF16 weight, with an F32 product.
#[test]
fn bf16_weight_matmul_decodes_the_weight_and_keeps_f32_precision() {
    let builder = Builder::new();
    let weight = builder.constant("table", TensorType::new([2, 3], DType::BF16));
    let x = builder.slot(Slot::Activation, TensorType::f32(vec![1, 3]));
    let transposed = builder.transpose(weight, vec![1, 0]);
    let logits = builder.matmul(x, transposed);
    let graph = builder.finish(logits);
    // 0.5 + 7/1024 is not a BF16 value, so narrowing the activation or the product would change the result.
    let activation = vec![0.5 + 7.0 / 1024.0, -0.5, 2.0];

    let output = eval(
        &graph,
        &HashMap::from([
            (weight.id, Value::from(table(&[2, 3]))),
            (
                x.id,
                Value::Host(HostTensor::f32(vec![1, 3], activation.clone())),
            ),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|r| r.output);
    // W = [[1, -2, 0.5], [3, -0.75, 1.5]], so x W^T is exact in f32:
    // row 0: 519/1024 + 1 + 1, row 1: 1557/1024 + 0.375 + 3.
    let expected = [2.0f32 + 519.0 / 1024.0, 3.375 + 1557.0 / 1024.0];
    let actual = dense_bits(&output.unwrap());
    assert_eq!(actual, expected.map(f32::to_bits));

    // A host BF16 weight of the same words gives the same bits as the owner-backed weight.
    let words = WORDS[..6].iter().map(|&(word, _)| word).collect();
    let dense: HashMap<_, Value> = HashMap::from([
        (weight.id, HostTensor::bf16(vec![2, 3], words).into()),
        (x.id, HostTensor::f32(vec![1, 3], activation).into()),
    ]);
    let dense = eval(&graph, &dense, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(dense_bits(&Value::Host(dense)), actual);
}

/// Card 380 FR-006. The same LM head after `fold_dense_contractions`: one `DenseContraction` reading
/// the weight in checkpoint `[N, K]` order with no transpose. The expected bits are the same hand-computed
/// literals as the matmul form, not compared against another poot evaluation.
#[test]
fn dense_contraction_matches_the_mixed_matmul_definition() {
    let builder = Builder::new();
    let weight = builder.constant("table", TensorType::new([2, 3], DType::BF16));
    let x = builder.slot(Slot::Activation, TensorType::f32(vec![1, 3]));
    let transposed = builder.transpose(weight, vec![1, 0]);
    let logits = builder.matmul(x, transposed);
    let graph = as_dense_contraction(builder.finish(logits), x.id, weight.id);

    assert!(
        graph.eqns.iter().any(|eqn| eqn.op
            == poot_graph_ir::OpKind::DenseContraction {
                weight: DType::BF16
            }),
        "the LM head must fold before this evaluates"
    );
    assert!(
        !graph
            .eqns
            .iter()
            .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::Transpose { .. })),
        "the folded graph holds no transpose to materialize"
    );

    // 0.5 + 7/1024 is not a BF16 value, so narrowing the activation or the product would change the result.
    let activation = vec![0.5 + 7.0 / 1024.0, -0.5, 2.0];
    let output = eval(
        &graph,
        &HashMap::from([
            (weight.id, Value::from(table(&[2, 3]))),
            (
                x.id,
                Value::Host(HostTensor::f32(vec![1, 3], activation.clone())),
            ),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|r| r.output);
    // W = [[1, -2, 0.5], [3, -0.75, 1.5]] in checkpoint order, so out[n] = sum_k x[k] * W[n, k]:
    // row 0: 519/1024 + 1 + 1, row 1: 1557/1024 + 0.375 + 3.
    let expected = [2.0f32 + 519.0 / 1024.0, 3.375 + 1557.0 / 1024.0];
    assert_eq!(dense_bits(&output.unwrap()), expected.map(f32::to_bits));
}

// Card 554d: `dense_eval_rejects_both_packed_bf16_readers_by_name` (Card 381 FR-007)
// deleted. Its premise - that `DenseContraction{weight:BF16}`/`DenseRowGather{source:BF16}` are a
// packed storage scheme the dense walk cannot read, distinct from the Spec 376 BF16 lane - no longer
// holds: `resolve::classify` admits exactly this `(F32, BF16, rank 2)` shape as
// `WeightContraction`/`RowGatherWiden` regardless of which pass produced the equation, and the one
// walk has no second, unsupported reading of the identical op/dtype/rank shape to reject. The two
// sibling tests this file already keeps - `dense_contraction_matches_the_mixed_matmul_definition` and
// `dense_row_gather_matches_gather_then_widen` - bind real data through these exact same constructed
// graphs and assert the correct computed bits, which is what the one walk actually does with them now.

/// Card 381 FR-006. The token embedding after `fold_dense_bf16_row_gathers`: one `DenseRowGather` in
/// place of the gather and widening cast. The expected bits are the decoded words
/// `bf16_gather_reads_the_selected_rows` asserts, not compared against another poot evaluation.
#[test]
fn dense_row_gather_matches_gather_then_widen() {
    let builder = Builder::new();
    let source = builder.constant("table", TensorType::new([4, 2], DType::BF16));
    let token = builder.slot(Slot::Token, TensorType::new([3], DType::I32));
    let rows = builder.gather(source, 0, token);
    let widened = builder.cast(rows, DType::F32);
    let graph = as_dense_row_gather(builder.finish(widened), source.id, token.id);

    assert!(
        graph.eqns.iter().any(|eqn| eqn.op
            == poot_graph_ir::OpKind::DenseRowGather {
                source: DType::BF16
            }),
        "the embedding must fold before this evaluates"
    );
    assert!(
        !graph
            .eqns
            .iter()
            .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::Gather { .. })),
        "the folded graph holds no BF16 gather to materialize"
    );

    let inputs = |ids: Vec<i32>| {
        HashMap::from([
            (source.id, Value::from(table(&[4, 2]))),
            (token.id, Value::Host(HostTensor::i32(vec![3], ids))),
        ])
    };
    let output = eval(
        &graph,
        &inputs(vec![2, 0, 3]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|r| r.output);
    let expected = [
        WORDS[4].1, WORDS[5].1, WORDS[0].1, WORDS[1].1, WORDS[6].1, WORDS[7].1,
    ];
    assert_eq!(dense_bits(&output.unwrap()), expected.map(f32::to_bits));

    // The bounds check is the gather's own, unchanged.
    for (ids, position, index, kind) in [
        (
            vec![1, 4, 0],
            1,
            4,
            crate::ops::index_rule::IndexFaultKind::OutOfRange,
        ),
        (
            vec![-1, 0, 0],
            0,
            -1,
            crate::ops::index_rule::IndexFaultKind::Negative,
        ),
    ] {
        let output = eval(
            &graph,
            &inputs(ids),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .map(|r| r.output);
        assert!(matches!(
            &output,
            Err(EvalError::Index(fault))
                if fault.position == position
                    && fault.value == crate::IndexValue::I32(index)
                    && fault.len == 4
                    && fault.kind == kind
        ));
    }

    // The Qwen3.8-27B chain has a reshape between the gather and the cast. The fold rebuilds it over F32,
    // so the same six values come back under the traced `[1, 3, 2]` shape.
    let builder = Builder::new();
    let source = builder.constant("table", TensorType::new([4, 2], DType::BF16));
    let token = builder.slot(Slot::Token, TensorType::new([3], DType::I32));
    let shaped = builder.reshape(builder.gather(source, 0, token), vec![1, 3, 2]);
    let widened = builder.cast(shaped, DType::F32);
    let mut graph = builder.finish(widened);
    // Hand-rewrite to the post-fold shape for a `Reshape` link (Card 381 FR-004): the row gather
    // writes the ungathered `[3, 2]` F32 rows at a fresh id, and the same reshape, rebuilt over F32,
    // reads it and writes `widened.id`.
    let gathered_f32 = graph.values.len();
    graph.values.push(poot_graph_ir::graph::ValueMeta::new(
        TensorType::f32(vec![3, 2]),
        poot_graph_ir::graph::Storage::Device,
        None,
    ));
    graph.eqns = vec![
        poot_graph_ir::Eqn {
            op: OpKind::DenseRowGather {
                source: DType::BF16,
            },
            inputs: vec![
                poot_graph_ir::Operand::Value(source.id),
                poot_graph_ir::Operand::Value(token.id),
            ],
            out: gathered_f32,
            layer: None,
        },
        poot_graph_ir::Eqn {
            op: OpKind::Reshape {
                shape: vec![1, 3, 2],
            },
            inputs: vec![poot_graph_ir::Operand::Value(gathered_f32)],
            out: widened.id,
            layer: None,
        },
    ];
    let output = eval(
        &graph,
        &HashMap::from([
            (source.id, Value::from(table(&[4, 2]))),
            (
                token.id,
                Value::Host(HostTensor::i32(vec![3], vec![2, 0, 3])),
            ),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|r| r.output);
    let output = output.unwrap();
    let Value::Host(tensor) = &output else {
        panic!("expected a dense output");
    };
    assert_eq!(tensor.shape(), vec![1, 3, 2]);
    assert_eq!(dense_bits(&output), expected.map(f32::to_bits));
}

#[test]
fn bf16_reshape_aliases_contiguous_views_only() {
    let builder = Builder::new();
    let source = builder.constant("table", TensorType::new([2, 3], DType::BF16));
    let reshaped = builder.reshape(source, vec![3, 2]);
    let graph = builder.finish(reshaped);
    let view = table(&[2, 3]);
    let output = eval(
        &graph,
        &HashMap::from([(source.id, Value::from(view))]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|r| r.output);
    let Value::Owner(ExactValue::Bf16(output)) = output.unwrap() else {
        panic!("reshape must stay exact BF16");
    };
    assert_eq!(output.shape(), [3, 2]);
    assert_eq!(
        (0..6).map(|flat| output.value(flat)).collect::<Vec<_>>(),
        WORDS[..6]
            .iter()
            .map(|&(_, value)| value)
            .collect::<Vec<_>>()
    );
    assert_eq!(output.physical_bytes(), 12);

    let builder = Builder::new();
    let source = builder.constant("table", TensorType::new([2, 3], DType::BF16));
    let reshaped = builder.reshape(builder.transpose(source, vec![1, 0]), vec![6]);
    let graph = builder.finish(reshaped);
    let output = eval(
        &graph,
        &HashMap::from([(source.id, Value::from(table(&[2, 3])))]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|r| r.output);
    assert!(matches!(
        output,
        Err(EvalError::ExactBf16(
            ExactBf16Error::NonContiguousReshape { .. }
        ))
    ));
}

// Card 554d: `bf16_admission_rejects_other_consumers_before_execution` deleted its
// `Slice`/`MatMul(BF16,BF16)`/`Cast(F32,BF16)` rejection assertions. Its premise - a closed list of
// Spec 376-admitted BF16-touching shapes, with every other shape rejected before any equation runs -
// no longer holds: `evaluate_dense`'s own `get()` closure (the "one-equation dense bridge", card
// 396) materializes an exact BF16 carrier through `dense_tensor_or_materialize` for ANY generic op
// that doesn't match a Spec 376 special case, one equation's worth at a time (safe, since nothing
// beyond that one equation's selected elements is ever materialized). `Slice` on a gathered BF16 row
// and `MatMul` of two derived-BF16 operands both now compute a real result through that bridge
// instead of refusing, and `Cast(F32, BF16)` was always an ordinary admitted narrowing cast
// (`evaluate_cast`'s `(F32, BF16|F16)` arm) - never gated by this admission list to begin with.
#[test]
fn matmul_rejects_bf16_weight_on_the_left_at_infer_time() {
    // Card 623 (SC-001): `MatMul` rejects the reverse pairing (BF16 activation x F32 weight,
    // "weight on the left") at `infer` now, so it never reaches the evaluator at all - a trace-time
    // `ShapeError`, not an `ExactBf16` admission rejection.
    let matmul_err = OpKind::MatMul
        .infer(&[
            TensorType::new(vec![2, 2], DType::BF16),
            TensorType::f32(vec![2, 2]),
        ])
        .expect_err("BF16 x F32 MatMul (weight on the left) must be rejected by infer");
    assert!(matches!(
        matmul_err,
        ShapeError::MatMulOperandDtype {
            a: DType::BF16,
            b: DType::F32
        }
    ));
}

/// `MatMul(F32, BF16)` admits an F32 operand, but an F32 owner view there is still a closed consumer.
#[test]
fn f32_owner_view_is_rejected_as_a_matmul_activation_before_execution() {
    let owners = owners(&[
        (
            "activation",
            "F32",
            &[1, 2],
            vec![0, 0, 0x80, 0x3f, 0, 0, 0, 0x40],
        ),
        (
            "table",
            "BF16",
            &[2, 3],
            bf16_bytes(WORDS[..6].iter().map(|&(w, _)| w)),
        ),
        ("other", "BF16", &[2], bf16_bytes([WORDS[6].0, WORDS[7].0])),
    ]);
    let view =
        |name: &str| Value::from(DenseOwnerTensorView::new(Arc::clone(&owners[name])).unwrap());
    let builder = Builder::new();
    let other = builder.constant("other", TensorType::new([2], DType::BF16));
    let activation = builder.constant("activation", TensorType::f32(vec![1, 2]));
    let weight = builder.constant("table", TensorType::new([2, 3], DType::BF16));
    // This widening runs first if preflight lets the F32 owner through, and records an allocation.
    let _earlier = builder.cast(other, DType::F32);
    let product = builder.matmul(activation, weight);
    let graph = builder.finish(product);
    let output = eval(
        &graph,
        &HashMap::from([
            (other.id, view("other")),
            (activation.id, view("activation")),
            (weight.id, view("table")),
        ]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .map(|r| r.output);
    assert!(
        matches!(
            output,
            Err(EvalError::ExactDense(ExactDenseError::GraphConsumer { value_id, .. }))
                if value_id == activation.id
        ),
        "{output:?}"
    );
}

// Card 554d: `bf16_reservation_failure_publishes_nothing` deleted. It forced a
// fallible BF16-table allocation failure via `observer.bf16_mut().force_reservation_failure()` and
// asserted `ExactBf16Error::Reservation`. Both the hook and the error variant are gone: BF16-table
// allocation in the new `walk.rs` uses `operand::initialized_arc_slice`, which is non-fallible, so
// there is no way to force this failure anymore. Not reinstated with an invented fallible-reservation
// hook (would require editing `walk.rs`/`operand.rs`, outside this file and outside a mechanical
// migration's authority).

/// SC-008 (second clause): an owner-backed BF16 table consumed by a non-`Gather` equation (the
/// widening `Cast(BF16, F32)`) above `EvalBudget`'s `max_alloc_bytes` is `EvalError::Budget`.
///
/// Mutation: drop the `opts.charge(...)` call in `walk.rs::evaluate_cast`'s `(DType::BF16,
/// DType::F32)` arm; the bounded call stops erroring and this test goes red.
#[test]
fn bf16_widening_cast_above_budget_is_refused() {
    let builder = Builder::new();
    let source = builder.constant("table", TensorType::new([2, 3], DType::BF16));
    let widened = builder.cast(source, DType::F32);
    let graph = builder.finish(widened);
    let inputs = HashMap::from([(source.id, Value::from(table(&[2, 3])))]);

    // 6 elements decode to 24 F32 bytes; a 20-byte ceiling must refuse before any byte is written.
    let bounded = eval(
        &graph,
        &inputs,
        EvalOptions::new(EvalBudget::bounded(1_000, 20)),
    )
    .expect_err("a BF16 table widened past max_alloc_bytes must refuse");
    assert!(
        matches!(
            bounded,
            EvalError::Budget {
                resource: "bytes",
                needed: 24,
                limit: 20,
                ..
            }
        ),
        "{bounded:?}"
    );

    // The same cast under UNBOUNDED evaluates (every other test in this file already relies on this;
    // restated here so the budget row is self-contained).
    let unbounded = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("UNBOUNDED must still evaluate the same cast");
    assert!(matches!(unbounded.output, Value::Host(_)));
}

/// SC-003: an F32 chain containing a BF16 `Gather`, an I32 `Binary` and a `Select`
/// evaluates to the same bits whether it stands alone or an unrelated `F32 -> E4M3FN -> F32` branch
/// is also present in the graph - admission depends only on each equation's own operands, never on
/// whether some OTHER value in the graph happens to be E4M3FN.
///
/// Mutation: reinstate a graph-wide scan that routes every equation to a closed op list whenever any
/// E4M3 value exists anywhere in the graph; the second graph (which has one) is refused, and this test
/// goes red.
#[test]
fn an_unrelated_e4m3fn_branch_does_not_change_an_otherwise_ordinary_mixed_dtype_chain() {
    use crate::fp8::encode_e4m3fn_tensor;

    // The shared BF16-gather + I32-binary/select + F32 chain both graphs build identically.
    let chain = |b: &Builder| -> (
        poot_graph_ir::Traced,
        poot_graph_ir::Traced,
        poot_graph_ir::Traced,
        poot_graph_ir::Traced,
        poot_graph_ir::Traced,
        poot_graph_ir::Traced,
    ) {
        let bf16_table = b.constant("table", TensorType::new([4, 2], DType::BF16));
        let idx = b.slot(Slot::Token, TensorType::new([2], DType::I32));
        let gathered = b.gather(bf16_table, 0, idx);
        let widened = b.cast(gathered, DType::F32);

        let cond = helpers::i32_constant(b, "cond", vec![2]).unwrap();
        let zero = helpers::i32_constant(b, "zero", vec![2]).unwrap();
        let one = helpers::i32_constant(b, "one", vec![2]).unwrap();
        let binary_cond = b.binary(poot_graph_ir::BinOp::GeU, cond, zero);
        let selected = b.select(binary_cond, one, zero);
        let selected_f32 = b.cast(selected, DType::F32);

        let sum = b.binary(poot_graph_ir::BinOp::Add, widened, selected_f32);
        (bf16_table, idx, cond, zero, one, sum)
    };

    let common_inputs =
        |bf16_table: ValueId, idx: ValueId, cond: ValueId, zero: ValueId, one: ValueId| {
            HashMap::from([
                (bf16_table, Value::from(table(&[4, 2]))),
                (idx, Value::Host(HostTensor::i32(vec![2], vec![1, 3]))),
                (cond, Value::Host(HostTensor::i32(vec![2], vec![-1, 0]))),
                (zero, Value::Host(HostTensor::i32(vec![2], vec![0, 0]))),
                (one, Value::Host(HostTensor::i32(vec![2], vec![1, 1]))),
            ])
        };

    // Graph A: the chain alone.
    let b = Builder::new();
    let (bf16_table, idx, cond, zero, one, sum) = chain(&b);
    let graph_a = b.finish(sum);
    let inputs_a = common_inputs(bf16_table.id, idx.id, cond.id, zero.id, one.id);
    let out_a = eval(&graph_a, &inputs_a, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("the chain alone must evaluate")
        .output;

    // Graph B: the same chain, plus an unrelated F32 -> E4M3FN -> F32 branch added into the same
    // output (0.0 round-trips through E4M3FN exactly, so it changes nothing numerically - only
    // whether the equation exists in the graph at all).
    let b = Builder::new();
    let (bf16_table, idx, cond, zero, one, sum) = chain(&b);
    let e4m3_in = b.constant("e4m3_in", TensorType::f32(vec![2]));
    let e4m3_mid = b.cast(e4m3_in, DType::E4M3FN);
    let e4m3_out = b.cast(e4m3_mid, DType::F32);
    let output_b = b.binary(poot_graph_ir::BinOp::Add, sum, e4m3_out);
    let graph_b = b.finish(output_b);
    let mut inputs_b = common_inputs(bf16_table.id, idx.id, cond.id, zero.id, one.id);
    inputs_b.insert(
        e4m3_in.id,
        Value::Host(HostTensor::f32(vec![2], vec![0.0, 0.0])),
    );
    let out_b = eval(&graph_b, &inputs_b, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("the chain plus an unrelated E4M3FN branch must also evaluate")
        .output;

    assert_eq!(dense_bits(&out_a), dense_bits(&out_b));

    // The unrelated branch really is E4M3FN, not silently elided: confirm the encode is exact zero,
    // so this isn't a vacuous "0.0 + 0.0" comparison.
    let zero_bytes = encode_e4m3fn_tensor(vec![2], &[0.0, 0.0]).unwrap();
    assert_eq!(zero_bytes.view().bytes(), &[0x00, 0x00]);
}
