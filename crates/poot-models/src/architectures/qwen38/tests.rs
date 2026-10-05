use super::*;
use poot_graph_ir::{OpKind, PackedSourceName, Storage};
use poot_quant::format::{ScaleEncoding, WeightFormat};
use poot_tensor::DType;
use poot_test_util::seed_of;
use std::collections::HashMap;

/// Card 534a: `fold_dense_bf16_row_gathers` moved into `poot-graph-plan`,
/// `pub(crate)` there, reachable only through `prepare_target_graph`. The PLE sharded-lookup graphs
/// below have no `MatMul`, no non-last-axis `Reduce`, and their E4M3FN table is untouched by the
/// backend dtype retype/widen step (which only ever retypes a `DType::BF16` const), so
/// `prepare_target_graph`'s other three steps are no-ops here and this is equivalent to the bare
/// fold. `Backend::SpirvVulkan` is an arbitrary pick - the fold itself is backend-neutral.
fn fold_dense_bf16_row_gathers<V: poot_graph_ir::ValidationChannel>(
    g: &poot_graph_ir::Graph<V>,
) -> poot_graph_ir::Graph<V> {
    poot_graph_plan::prepare_target_graph(
        g,
        poot_target::Backend::SpirvVulkan,
        &poot_test_util::device_caps::default_caps_for(poot_target::Backend::SpirvVulkan),
    )
}

fn tiny_cfg() -> Qwen4ExpConfig {
    Qwen4ExpConfig {
        hidden: 8,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 4,
        rotary_dim: 2,
        eps: 1e-5,
    }
}

fn tiny_qcfg(index_budget: usize) -> QsaConfig {
    QsaConfig {
        index_n_heads: 2,
        index_kv_heads: 1,
        index_head_dim: 3,
        index_budget,
        index_compress_ratio: 2,
    }
}

#[test]
fn qwen38_fp8_projection_table_matches_official_split_names_and_shapes() {
    let got: Vec<_> = Qwen4ExpExpertProjection::ALL
        .into_iter()
        .map(|role| {
            (
                role.checkpoint_prefix(47, 511),
                role.graph_prefix(47, 511),
                role.out_in(2560, 640),
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![
            (
                "model.language_model.layers.47.mlp.experts.511.gate_proj".to_string(),
                "layers.47.mlp.experts.511.gate_proj".to_string(),
                (640, 2560),
            ),
            (
                "model.language_model.layers.47.mlp.experts.511.up_proj".to_string(),
                "layers.47.mlp.experts.511.up_proj".to_string(),
                (640, 2560),
            ),
            (
                "model.language_model.layers.47.mlp.experts.511.down_proj".to_string(),
                "layers.47.mlp.experts.511.down_proj".to_string(),
                (2560, 640),
            ),
        ]
    );
}

mod packed_routed {
    use std::sync::Arc;

    use poot_eval::{EvalBudget, EvalOptions, Value, eval};
    use poot_graph_ir::op::PackedWeight;
    use poot_quant::{OperandRole, PackedComponentRef, PackedPayload, SourceRole};
    use poot_tensor::HostTensor;

    use super::*;

    fn descriptor(
        role: Qwen4ExpExpertProjection,
        hidden: usize,
        intermediate: usize,
    ) -> PackedWeight {
        let (out, k) = role.out_in(hidden, intermediate);
        PackedWeight::try_new(
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::Bf16,
            },
            [out, k],
        )
        .expect("bounded packed descriptor")
    }

    fn scale_elements(descriptor: PackedWeight) -> usize {
        let bytes_per_element = descriptor
            .format()
            .descriptor()
            .planar_operand(OperandRole::Scale)
            .expect("a packed weight has a Scale operand")
            .element_bytes();
        descriptor.source_bytes(SourceRole::Planar(OperandRole::Scale)) / bytes_per_element
    }

    fn packed_tables(
        layer: usize,
        experts: usize,
        hidden: usize,
        intermediate: usize,
    ) -> (
        Qwen4ExpPackedExpertTables,
        HashMap<String, Arc<PackedPayload>>,
    ) {
        let mut owners = HashMap::new();
        let tables = Qwen4ExpExpertProjection::ALL
            .into_iter()
            .map(|role| {
                let descriptor = descriptor(role, hidden, intermediate);
                let rows = (0..experts)
                    .map(|expert| {
                        let linear_id = role.checkpoint_prefix(layer, expert);
                        let code = match role {
                            Qwen4ExpExpertProjection::Gate | Qwen4ExpExpertProjection::Down => 0x38,
                            Qwen4ExpExpertProjection::Up => [0x38, 0x40, 0x44, 0x48][expert],
                        };
                        let weights = vec![
                            code;
                            descriptor
                                .source_bytes(SourceRole::Planar(OperandRole::Codes))
                        ];
                        let scales = (0..scale_elements(descriptor))
                            .flat_map(|_| 0x3f80u16.to_le_bytes())
                            .collect::<Vec<_>>();
                        let owner = Arc::new(
                            PackedPayload::try_new(
                                descriptor,
                                [
                                    (SourceRole::Planar(OperandRole::Codes), weights.into()),
                                    (SourceRole::Planar(OperandRole::Scale), scales.into()),
                                ],
                            )
                            .expect("bounded packed payload"),
                        );
                        assert!(owners.insert(linear_id.clone(), owner).is_none());
                        PackedLinearGraphRow {
                            ordinal: expert,
                            linear_id,
                            descriptor,
                        }
                    })
                    .collect();
                Qwen4ExpPackedProjectionTable::new(role, rows)
            })
            .collect();
        (
            Qwen4ExpPackedExpertTables::try_new(tables, layer, experts, hidden, intermediate)
                .expect("valid packed tables"),
            owners,
        )
    }

    #[derive(Clone, Copy, Debug)]
    enum TableMutation {
        SwapRoles,
        WrongExpertCount,
        WrongOrdinal,
        WrongLayerId,
        SameRoleExpertSwap,
        GateUpIdSwap,
        WrongFormat,
        WrongShape,
    }

    #[test]
    fn qwen4exp_packed_projection_tables_reject_role_and_row_mutations() {
        let layer = 7;
        let (valid, _) = packed_tables(layer, 4, 4, 1);
        let cases = [
            TableMutation::SwapRoles,
            TableMutation::WrongExpertCount,
            TableMutation::WrongOrdinal,
            TableMutation::WrongLayerId,
            TableMutation::SameRoleExpertSwap,
            TableMutation::GateUpIdSwap,
            TableMutation::WrongFormat,
            TableMutation::WrongShape,
        ];
        for mutation in cases {
            let mut tables = valid.tables.clone();
            match mutation {
                TableMutation::SwapRoles => tables.swap(0, 1),
                TableMutation::WrongExpertCount => {
                    tables[Qwen4ExpExpertProjection::Down.ordinal()].rows.pop();
                }
                TableMutation::WrongOrdinal => {
                    tables[Qwen4ExpExpertProjection::Up.ordinal()].rows[2].ordinal = 1;
                }
                TableMutation::WrongLayerId => {
                    tables[Qwen4ExpExpertProjection::Gate.ordinal()].rows[0].linear_id =
                        Qwen4ExpExpertProjection::Gate.checkpoint_prefix(layer + 1, 0);
                }
                TableMutation::SameRoleExpertSwap => {
                    let down = Qwen4ExpExpertProjection::Down.ordinal();
                    tables[down].rows.swap(1, 2);
                    for (expert, row) in tables[down].rows.iter_mut().enumerate() {
                        row.ordinal = expert;
                    }
                }
                TableMutation::GateUpIdSwap => {
                    let gate = Qwen4ExpExpertProjection::Gate.ordinal();
                    let up = Qwen4ExpExpertProjection::Up.ordinal();
                    let gate_id = tables[gate].rows[3].linear_id.clone();
                    let up_id = tables[up].rows[3].linear_id.clone();
                    tables[gate].rows[3].linear_id = up_id;
                    tables[up].rows[3].linear_id = gate_id;
                }
                TableMutation::WrongFormat => {
                    tables[Qwen4ExpExpertProjection::Gate.ordinal()].rows[0].descriptor =
                        PackedWeight::try_new(
                            WeightFormat::E4m3Block128 {
                                scale: ScaleEncoding::F32,
                            },
                            [1, 4],
                        )
                        .unwrap();
                }
                TableMutation::WrongShape => {
                    tables[Qwen4ExpExpertProjection::Up.ordinal()].rows[1].descriptor =
                        PackedWeight::try_new(
                            WeightFormat::E4m3Block128 {
                                scale: ScaleEncoding::Bf16,
                            },
                            [2, 4],
                        )
                        .unwrap();
                }
            }
            let error = Qwen4ExpPackedExpertTables::try_new(tables, layer, 4, 4, 1)
                .expect_err("mutated table must be rejected");
            let distinguished = matches!(
                (mutation, error),
                (
                    TableMutation::SwapRoles,
                    Qwen4ExpPackedExpertTableError::ProjectionRole { .. }
                ) | (
                    TableMutation::WrongExpertCount,
                    Qwen4ExpPackedExpertTableError::ExpertCount { .. }
                ) | (
                    TableMutation::WrongOrdinal,
                    Qwen4ExpPackedExpertTableError::ExpertOrdinal { .. }
                ) | (
                    TableMutation::WrongLayerId,
                    Qwen4ExpPackedExpertTableError::LinearId { .. }
                ) | (
                    TableMutation::SameRoleExpertSwap,
                    Qwen4ExpPackedExpertTableError::LinearId { .. }
                ) | (
                    TableMutation::GateUpIdSwap,
                    Qwen4ExpPackedExpertTableError::LinearId { .. }
                ) | (
                    TableMutation::WrongFormat,
                    Qwen4ExpPackedExpertTableError::Format { .. }
                ) | (
                    TableMutation::WrongShape,
                    Qwen4ExpPackedExpertTableError::LogicalShape { .. }
                )
            );
            assert!(
                distinguished,
                "{mutation:?} did not produce its distinct typed error"
            );
        }
    }

    fn routed_inputs(
        b: &Builder,
        router_shape: Vec<usize>,
        shared_down_shape: Vec<usize>,
    ) -> [Traced; 7] {
        [
            b.constant("probe.x", TensorType::f32(vec![1, 4, 4])),
            b.constant("probe.router", TensorType::f32(router_shape)),
            b.constant("probe.shared_gate", TensorType::f32(vec![4, 1])),
            b.constant("probe.shared_up", TensorType::f32(vec![4, 1])),
            b.constant("probe.shared_down", TensorType::f32(shared_down_shape)),
            b.constant("probe.shared_weight", TensorType::f32(vec![4, 1])),
            b.constant("probe.sentinel", TensorType::f32(vec![1])),
        ]
    }

    #[derive(Clone, Copy, Debug)]
    enum PreflightMutation {
        RouterShape,
        SharedShape,
        UpWeightCollision,
        DownScaleCollision,
    }

    #[test]
    fn qwen4exp_packed_builder_rejects_before_mutating_any_collection() {
        let (tables, _) = packed_tables(7, 4, 4, 1);
        for mutation in [
            PreflightMutation::RouterShape,
            PreflightMutation::SharedShape,
            PreflightMutation::UpWeightCollision,
            PreflightMutation::DownScaleCollision,
        ] {
            let b = Builder::new();
            let router_shape = if matches!(mutation, PreflightMutation::RouterShape) {
                vec![4, 3]
            } else {
                vec![4, 4]
            };
            let shared_down_shape = if matches!(mutation, PreflightMutation::SharedShape) {
                vec![2, 4]
            } else {
                vec![1, 4]
            };
            let [
                x,
                router,
                shared_gate,
                shared_up,
                shared_down,
                shared_weight,
                sentinel,
            ] = routed_inputs(&b, router_shape, shared_down_shape);
            let collision_name = match mutation {
                PreflightMutation::UpWeightCollision => Some(PackedSourceName::weight(
                    &tables.projection(Qwen4ExpExpertProjection::Up).rows()[3].linear_id,
                )),
                PreflightMutation::DownScaleCollision => Some(PackedSourceName::scale(
                    &tables.projection(Qwen4ExpExpertProjection::Down).rows()[3].linear_id,
                )),
                PreflightMutation::RouterShape | PreflightMutation::SharedShape => None,
            };
            if let Some(name) = collision_name.as_ref() {
                b.constant(name.as_str(), TensorType::new(vec![1, 1], DType::I8));
            }
            let generation = b.generation();
            let error = qwen38_packed_indexed_moe_ffn(
                &b,
                x,
                router,
                shared_gate,
                shared_up,
                shared_down,
                shared_weight,
                2,
                &tables,
            )
            .expect_err("preflight mutation must fail");
            assert_eq!(
                b.generation(),
                generation,
                "{mutation:?} changed generation"
            );
            let expected_error = match (mutation, error) {
                (
                    PreflightMutation::RouterShape,
                    Qwen4ExpPackedRoutedError::InputDimension {
                        field: "router_w",
                        axis: 1,
                        actual: 3,
                        expected: 4,
                    },
                )
                | (
                    PreflightMutation::SharedShape,
                    Qwen4ExpPackedRoutedError::InputDimension {
                        field: "shexp_down_w",
                        axis: 0,
                        actual: 2,
                        expected: 1,
                    },
                ) => true,
                (
                    PreflightMutation::UpWeightCollision | PreflightMutation::DownScaleCollision,
                    Qwen4ExpPackedRoutedError::Graph(BuilderAppendError::NameCollision {
                        name,
                        requested: poot_graph_ir::BuilderValueNamespace::Constant,
                        existing: poot_graph_ir::BuilderValueNamespace::Constant,
                    }),
                ) => collision_name.as_ref().map(PackedSourceName::as_str) == Some(name.as_str()),
                _ => false,
            };
            assert!(
                expected_error,
                "{mutation:?} returned the wrong typed error"
            );

            let graph = b.finish(sentinel);
            assert_eq!(graph.values.len(), generation as usize);
            assert_eq!(graph.inputs.len(), generation as usize);
            assert_eq!(graph.consts.len(), generation as usize);
            assert!(graph.slots.is_empty());
            assert!(graph.eqns.is_empty(), "{mutation:?} appended equations");
        }
    }

    fn dense_probe(hidden: usize, experts: usize, rows: usize) -> Graph {
        let b = Builder::new();
        let x = b.constant("probe.x", TensorType::f32(vec![1, rows, hidden]));
        let output = qwen38_moe_ffn(&b, x, "probe", hidden, experts, 2, 1, 1);
        b.finish(output)
    }

    fn packed_probe(
        hidden: usize,
        experts: usize,
        rows: usize,
        tables: &Qwen4ExpPackedExpertTables,
    ) -> Graph {
        let b = Builder::new();
        let x = b.constant("probe.x", TensorType::f32(vec![1, rows, hidden]));
        let router = b.constant(
            "probe.mlp.gate.weight",
            TensorType::f32(vec![hidden, experts]),
        );
        let shared_gate = b.constant(
            "probe.mlp.shared_expert.gate_proj.weight",
            TensorType::f32(vec![hidden, 1]),
        );
        let shared_up = b.constant(
            "probe.mlp.shared_expert.up_proj.weight",
            TensorType::f32(vec![hidden, 1]),
        );
        let shared_down = b.constant(
            "probe.mlp.shared_expert.down_proj.weight",
            TensorType::f32(vec![1, hidden]),
        );
        let shared_weight = b.constant(
            "probe.mlp.shared_expert_gate.weight",
            TensorType::f32(vec![hidden, 1]),
        );
        let output = qwen38_packed_indexed_moe_ffn(
            &b,
            x,
            router,
            shared_gate,
            shared_up,
            shared_down,
            shared_weight,
            2,
            tables,
        )
        .expect("packed indexed routed graph");
        b.finish(output)
    }

    fn packed_grouped_probe(
        hidden: usize,
        experts: usize,
        rows: usize,
        tables: &Qwen4ExpPackedExpertTables,
    ) -> Graph {
        let top_k = 2;
        let b = Builder::new();
        let x = b.constant("probe.x", TensorType::f32(vec![1, rows, hidden]));
        let router = b.constant(
            "probe.mlp.gate.weight",
            TensorType::f32(vec![hidden, experts]),
        );
        let shared_gate = b.constant(
            "probe.mlp.shared_expert.gate_proj.weight",
            TensorType::f32(vec![hidden, 1]),
        );
        let shared_up = b.constant(
            "probe.mlp.shared_expert.up_proj.weight",
            TensorType::f32(vec![hidden, 1]),
        );
        let shared_down = b.constant(
            "probe.mlp.shared_expert.down_proj.weight",
            TensorType::f32(vec![1, hidden]),
        );
        let shared_weight = b.constant(
            "probe.mlp.shared_expert_gate.weight",
            TensorType::f32(vec![hidden, 1]),
        );
        let output = qwen38_packed_grouped_moe_ffn(
            &b,
            x,
            router,
            shared_gate,
            shared_up,
            shared_down,
            shared_weight,
            top_k,
            tables,
        )
        .expect("packed grouped routed graph");
        b.finish(output)
    }

    fn dense_expert_gate_up(hidden: usize, coefficients: &[f32]) -> Vec<f32> {
        coefficients
            .iter()
            .flat_map(|&coefficient| (0..hidden).flat_map(move |_| [1.0, coefficient]))
            .collect()
    }

    fn dense_expert_down(hidden: usize, experts: usize) -> Vec<f32> {
        vec![1.0; experts * hidden]
    }

    fn fixture_dense_value(name: &str, shape: &[usize], route_ids: &[usize]) -> Vec<f32> {
        let hidden = 4;
        let experts = 4;
        let coefficients = [1.0, 2.0, 3.0, 4.0];
        match name {
            "probe.x" => route_ids
                .iter()
                .flat_map(|&expert| {
                    (0..hidden).map(move |column| if column == expert { 1.0 } else { 0.0 })
                })
                .collect(),
            "probe.mlp.gate.weight" => (0..hidden)
                .flat_map(|row| {
                    (0..experts).map(move |column| if row == column { 1.0 } else { 0.0 })
                })
                .collect(),
            "probe.mlp.experts.gate_up_proj" => dense_expert_gate_up(hidden, &coefficients),
            "probe.mlp.experts.down_proj" => dense_expert_down(hidden, experts),
            "probe.mlp.shared_expert.gate_proj.weight" => vec![1.0; hidden],
            "probe.mlp.shared_expert.up_proj.weight" => vec![2.0; hidden],
            "probe.mlp.shared_expert.down_proj.weight" => vec![1.0; hidden],
            "probe.mlp.shared_expert_gate.weight" => vec![1.0; hidden],
            _ => panic!("unexpected routed fixture input {name} with shape {shape:?}"),
        }
    }

    fn eval_probe(
        graph: &Graph,
        owners: &HashMap<String, Arc<PackedPayload>>,
        route_ids: &[usize],
    ) -> Vec<f32> {
        graph.validate().expect("valid routed graph");
        let mut inputs = HashMap::new();
        for &id in &graph.inputs {
            let meta = graph.meta(id);
            let name = meta.name.as_deref().expect("named routed probe input");
            let value = if let Some(source) = PackedSourceName::parse(name) {
                Value::Packed(PackedComponentRef::new(
                    Arc::clone(&owners[source.linear_id()]),
                    source.role(),
                ))
            } else {
                Value::Host(HostTensor::f32(
                    meta.aval.shape.clone(),
                    fixture_dense_value(name, &meta.aval.shape, route_ids),
                ))
            };
            inputs.insert(id, value);
        }
        let Value::Host(output) = eval(graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("bounded routed differential oracle")
            .output
        else {
            panic!("routed output must be dense")
        };
        output.as_f32().unwrap().to_vec()
    }

    #[derive(Clone, Copy)]
    enum OracleMutation {
        None,
        NoRenormalization,
        HigherIdWinsTie,
        OmitSharedExpert,
    }

    fn routed_reference(route_ids: &[usize], mutation: OracleMutation) -> Vec<f32> {
        let hidden = 4;
        let experts = 4;
        let coefficients = [1.0f32, 2.0, 3.0, 4.0];
        let silu = |value: f32| value / (1.0 + (-value).exp());
        let shared = if matches!(mutation, OracleMutation::OmitSharedExpert) {
            0.0
        } else {
            let shared_gate = 1.0 / (1.0 + (-1.0f32).exp());
            shared_gate * silu(1.0) * 2.0
        };

        route_ids
            .iter()
            .flat_map(|&primary| {
                let logits = (0..experts)
                    .map(|expert| if expert == primary { 1.0f32 } else { 0.0f32 })
                    .collect::<Vec<_>>();
                let mut order = (0..experts).collect::<Vec<_>>();
                order.sort_by(|&left, &right| {
                    logits[right]
                        .partial_cmp(&logits[left])
                        .unwrap()
                        .then_with(|| {
                            if matches!(mutation, OracleMutation::HigherIdWinsTie) {
                                right.cmp(&left)
                            } else {
                                left.cmp(&right)
                            }
                        })
                });
                let selected = &order[..2];
                let exponent = |expert: usize| (logits[expert] - 1.0).exp();
                let denominator: f32 = if matches!(mutation, OracleMutation::NoRenormalization) {
                    (0..experts).map(exponent).sum()
                } else {
                    selected.iter().copied().map(exponent).sum()
                };
                let routed = selected
                    .iter()
                    .copied()
                    .map(|expert| exponent(expert) / denominator * silu(1.0) * coefficients[expert])
                    .sum::<f32>();
                vec![routed + shared; hidden]
            })
            .collect()
    }

    fn assert_distinct(actual: &[f32], mutated: &[f32], context: &str) {
        assert!(
            poot_test_util::max_abs_error(actual, mutated) > 0.1,
            "{context} mutation did not change the oracle"
        );
    }

    #[test]
    fn qwen4exp_packed_and_dense_sources_share_router_semantics() {
        let (hidden, experts, rows) = (4, 4, 4);
        let route_ids = [2usize, 0, 2, 1];
        let (tables, owners) = packed_tables(7, experts, hidden, 1);
        let dense = dense_probe(hidden, experts, rows);
        let packed = packed_probe(hidden, experts, rows, &tables);

        assert_eq!(
            packed
                .eqns
                .iter()
                .filter(|eqn| matches!(eqn.op, OpKind::PackedDequant { .. }))
                .count(),
            Qwen4ExpExpertProjection::ALL.len() * experts
        );
        assert!(!packed.consts.iter().any(|&id| {
            packed
                .meta(id)
                .name
                .as_deref()
                .is_some_and(|name| name.contains("gate_up_proj"))
        }));
        let dense_output = eval_probe(&dense, &HashMap::new(), &route_ids);
        let packed_output = eval_probe(&packed, &owners, &route_ids);
        let expected = routed_reference(&route_ids, OracleMutation::None);
        // dense versus packed
        poot_test_util::assert_close(&dense_output, &packed_output, 1e-5);
        // packed versus the independent reference
        poot_test_util::assert_close(&packed_output, &expected, 1e-5);

        let mut swapped_owners = owners.clone();
        let first_id = Qwen4ExpExpertProjection::Up.checkpoint_prefix(7, 0);
        let second_id = Qwen4ExpExpertProjection::Up.checkpoint_prefix(7, 2);
        let first = swapped_owners
            .remove(&first_id)
            .expect("first ordered packed source");
        let second = swapped_owners
            .remove(&second_id)
            .expect("second ordered packed source");
        assert!(swapped_owners.insert(first_id, second).is_none());
        assert!(swapped_owners.insert(second_id, first).is_none());
        let swapped_output = eval_probe(&packed, &swapped_owners, &route_ids);
        assert_distinct(&packed_output, &swapped_output, "source ordering");

        for (mutation, context) in [
            (OracleMutation::NoRenormalization, "renormalization"),
            (OracleMutation::HigherIdWinsTie, "stable tie choice"),
            (OracleMutation::OmitSharedExpert, "shared expert addition"),
        ] {
            assert_distinct(
                &packed_output,
                &routed_reference(&route_ids, mutation),
                context,
            );
        }
    }

    /// Card 362 FR-007/SC-003: the same `route_ids` as the indexed test select expert 0 four times,
    /// experts 1 and 2 twice each, and expert 3 never: one empty group and unequal nonempty groups over
    /// `rows * top_k = 8` activation rows, sorted by expert before the packed matmul and restored to
    /// row order after. A grouped composition that assumed equal groups or dropped the inverse gather
    /// would scramble which output row belongs to which input token; per-token comparison against the
    /// indexed packed path and the independent reference catches that.
    #[test]
    fn qwen4exp_grouped_routes_restore_row_order() {
        let (hidden, experts, rows) = (4, 4, 4);
        let route_ids = [2usize, 0, 2, 1];
        let (tables, owners) = packed_tables(7, experts, hidden, 1);
        let indexed = packed_probe(hidden, experts, rows, &tables);
        let grouped = packed_grouped_probe(hidden, experts, rows, &tables);

        let group_sizes = {
            let mut counts = [0usize; 4];
            for &primary in &route_ids {
                let mut order = (0..experts).collect::<Vec<_>>();
                order.sort_by_key(|&candidate| (candidate != primary, candidate));
                for &expert in &order[..2] {
                    counts[expert] += 1;
                }
            }
            counts
        };
        assert!(
            group_sizes.contains(&0),
            "fixture must exercise an empty group: {group_sizes:?}"
        );
        let mut distinct_nonzero = group_sizes
            .iter()
            .copied()
            .filter(|&count| count > 0)
            .collect::<Vec<_>>();
        distinct_nonzero.sort_unstable();
        distinct_nonzero.dedup();
        assert!(
            distinct_nonzero.len() > 1,
            "fixture must exercise unequal nonempty groups: {group_sizes:?}"
        );

        let indexed_output = eval_probe(&indexed, &owners, &route_ids);
        let grouped_output = eval_probe(&grouped, &owners, &route_ids);
        let expected = routed_reference(&route_ids, OracleMutation::None);
        // grouped versus the independent reference
        poot_test_util::assert_close(&grouped_output, &expected, 1e-5);
        // indexed versus grouped row order
        poot_test_util::assert_close(&indexed_output, &grouped_output, 1e-5);

        for (mutation, context) in [
            (OracleMutation::NoRenormalization, "renormalization"),
            (OracleMutation::HigherIdWinsTie, "stable tie choice"),
            (OracleMutation::OmitSharedExpert, "shared expert addition"),
        ] {
            assert_distinct(
                &grouped_output,
                &routed_reference(&route_ids, mutation),
                context,
            );
        }
    }
}

#[test]
fn trace_qwen38_qsa_probe_validates() {
    let cfg = tiny_cfg();
    let qcfg = tiny_qcfg(4); // block_topk = 2
    let g = trace_qwen38_qsa_probe(cfg, qcfg, 2, 8);
    g.validate()
        .expect("qwen38 qsa probe graph should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 8, cfg.hidden]);
    assert_eq!(g.state.len(), 2 * 2, "k_cache+v_cache per layer");
}

#[test]
fn block_topk_matches_survey_derivation() {
    // spec 282: "512 blocks or 2048 tokens" is one budget, not two knobs.
    let real = QsaConfig {
        index_n_heads: 4,
        index_kv_heads: 1,
        index_head_dim: 128,
        index_budget: 2048,
        index_compress_ratio: 4,
    };
    assert_eq!(real.block_topk(), 512);
}

#[test]
fn tail_mask_matches_hand_derivation_l8_c2() {
    // L=8, C=2: block boundaries at 2,4,6,8. Hand-checked expectations (spec 282 FR-003):
    // i=0 (block0 in progress): tail = {0}.
    // i=1 (block0 just completed): tail = {}.
    // i=2 (block1 in progress): tail = {2}.
    // i=4 (block2 in progress): tail = {4}.
    // i=7 (block3 just completed): tail = {}.
    let m = tail_mask_data(8, 2);
    let visible = |i: usize| -> Vec<usize> { (0..8).filter(|&j| m[i * 8 + j] == 0.0).collect() };
    assert_eq!(visible(0), vec![0]);
    assert_eq!(visible(1), Vec::<usize>::new());
    assert_eq!(visible(2), vec![2]);
    assert_eq!(visible(4), vec![4]);
    assert_eq!(visible(7), Vec::<usize>::new());
}

#[test]
fn block_eligible_matches_hand_derivation_l8_c2() {
    // Block b eligible at query i iff b*2+1 <= i. Block0(pos0-1): eligible from i=1. Block1(2-3):
    // from i=3. Block2(4-5): from i=5. Block3(6-7): from i=7.
    let m = block_eligible_mask_data(8, 4, 2);
    let eligible = |i: usize| -> Vec<usize> { (0..4).filter(|&b| m[i * 4 + b] == 0.0).collect() };
    assert_eq!(eligible(0), Vec::<usize>::new());
    assert_eq!(eligible(1), vec![0]);
    assert_eq!(eligible(3), vec![0, 1]);
    assert_eq!(eligible(5), vec![0, 1, 2]);
    assert_eq!(eligible(7), vec![0, 1, 2, 3]);
}

/// Bind every `Const` of a mask-only graph by name from `data` and return the tokens each query
/// row may attend to (mask value `0.0`), `row_len` tokens per row.
fn visible_tokens(g: &Graph, data: &HashMap<&str, Vec<f32>>, row_len: usize) -> Vec<Vec<usize>> {
    let inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = g
        .inputs
        .iter()
        .map(|&id| {
            let meta = g.meta(id);
            let name = meta.name.as_deref().expect("const without a name");
            let values = data
                .get(name)
                .cloned()
                .unwrap_or_else(|| panic!("no data bound for {name}"));
            (
                id,
                poot_tensor::HostTensor::f32(meta.aval.shape.clone(), values).into(),
            )
        })
        .collect();
    let mask = poot_eval::eval(
        g,
        &inputs,
        poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
    )
    .expect("cpu eval")
    .output
    .into_host()
    .expect("dense output")
    .as_f32()
    .unwrap()
    .to_vec();
    mask.chunks(row_len)
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|&(_, &v)| v == 0.0)
                .map(|(j, _)| j)
                .collect()
        })
        .collect()
}

/// Card 609: QSA's block top-k keeps exactly `block_topk` blocks when index scores tie, and breaks the
/// tie toward the lower block index. `L=4`, `C=2`, `block_topk=1`, every score `0.0` (the ReLU floor, where
/// real indexer scores tie whenever every head's dot product is negative). At query 3 both blocks are
/// complete and tied: only block 0 may be visible. A shared-rank top-k admits both, attending to 4 tokens on
/// a 2-token budget. Checks the prefill mask at every query and the decode mask at `pos=3`.
#[test]
fn qsa_block_top_k_keeps_exactly_the_budget_on_tied_scores() {
    let (l, c, block_topk) = (4, 2, 1);
    let nb = l / c;

    let b = Builder::new();
    let scores = b.constant("scores", TensorType::f32(vec![1, 1, l, nb]));
    let eligible = b.constant("qsa.block_eligible", TensorType::f32(vec![1, 1, l, nb]));
    let causal = b.constant("causal.mask", TensorType::f32(vec![1, 1, l, l]));
    let tail = b.constant("qsa.tail_mask", TensorType::f32(vec![1, 1, l, l]));
    let mask = qsa_combined_mask(&b, scores, eligible, causal, tail, block_topk, l, nb, c);
    let prefill = b.finish(mask);
    let prefill_data = HashMap::from([
        ("scores", vec![0.0; l * nb]),
        ("qsa.block_eligible", block_eligible_mask_data(l, nb, c)),
        ("causal.mask", causal_mask_data(l)),
        ("qsa.tail_mask", tail_mask_data(l, c)),
    ]);
    assert_eq!(
        visible_tokens(&prefill, &prefill_data, l),
        vec![vec![0], vec![0, 1], vec![0, 1, 2], vec![0, 1]],
        "prefill QSA mask per query: query 3 must keep only block 0 of the two tied blocks"
    );

    // Decode at pos=3 (the query that closes block 1): both blocks eligible, no tail block.
    let b = Builder::new();
    let scores = b.constant("scores", TensorType::f32(vec![1, 1, 1, nb]));
    let eligible = b.constant("qsa.block_eligible", TensorType::f32(vec![1, 1, 1, nb]));
    let tail_block = b.constant("qsa.tail_block", TensorType::f32(vec![1, 1, 1, nb]));
    let causal = b.constant("causal.mask", TensorType::f32(vec![1, 1, 1, l]));
    let mask = qsa_combined_mask_decode(
        &b, scores, eligible, tail_block, causal, block_topk, nb, c, l,
    );
    let decode = b.finish(mask);
    let decode_data = HashMap::from([
        ("scores", vec![0.0; nb]),
        ("qsa.block_eligible", vec![0.0; nb]),
        ("qsa.tail_block", vec![QSA_MASK_NEG; nb]),
        ("causal.mask", vec![0.0; l]),
    ]);
    assert_eq!(
        visible_tokens(&decode, &decode_data, l),
        vec![vec![0, 1]],
        "decode QSA mask at pos=3: only block 0 of the two tied blocks"
    );
}

/// SC-003: a QSA (full-attention) layer and Gated-DeltaNet (linear-attention) layers compose in one
/// `Graph`, following the `layer_types` schedule shape (3 linear layers then 1 full layer, spec 282
/// FR-006). Structural only: GDN numerics are covered by `crate::qwen3next`'s tests, so this checks
/// wiring (`validate()` succeeds, output shape matches). Reuses
/// `crate::qwen3next::qwen3next_gdn_prefill_block` unchanged and `deepseek2_dense_ffn` as the FFN
/// stand-in.
#[test]
fn mixed_gdn_qsa_stack_validates_sc003() {
    use crate::deepseek2::deepseek2_dense_ffn;
    use crate::qwen3next::qwen3next_gdn_prefill_block;
    use poot_graph_ir::Slot;

    let cfg = tiny_cfg();
    let qcfg = tiny_qcfg(4);
    let l = 8;
    let vocab = 6;
    let (gk, gv, ghd, gck, chunk) = (1usize, 2usize, 2usize, 2usize, 4usize);
    let h = cfg.hidden;

    let b = Builder::new();
    let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
    let embed = b.constant("model.embed_tokens.weight", TensorType::f32(vec![vocab, h]));
    let emb = b.gather(embed, 0, tokens);
    let mut x = b.reshape(emb, vec![1, l, h]);

    let cos = b.constant("rope.cos", TensorType::f32(vec![l, cfg.rotary_dim]));
    let sin = b.constant("rope.sin", TensorType::f32(vec![l, cfg.rotary_dim]));
    let nb = l / qcfg.index_compress_ratio;
    let block_cos = b.constant(
        "qsa.block_rope.cos",
        TensorType::f32(vec![nb, cfg.rotary_dim]),
    );
    let block_sin = b.constant(
        "qsa.block_rope.sin",
        TensorType::f32(vec![nb, cfg.rotary_dim]),
    );
    let causal_mask = b.constant("causal.mask", TensorType::f32(vec![1, 1, l, l]));
    let block_eligible = b.constant("qsa.block_eligible", TensorType::f32(vec![1, 1, l, nb]));
    let tail_mask = b.constant("qsa.tail_mask", TensorType::f32(vec![1, 1, l, l]));
    let tril_incl = b.constant("gdn.tril_incl", TensorType::f32(vec![1, 1, chunk, chunk]));
    let tril_strict = b.constant("gdn.tril_strict", TensorType::f32(vec![1, 1, chunk, chunk]));

    // The `layer_types` shape: N linear layers then 1 full (QSA) layer, repeating.
    let layer_is_full = [false, false, false, true];
    let mut state: Vec<(Traced, Traced)> = Vec::new();
    for (li, &is_full) in layer_is_full.iter().enumerate() {
        let p = format!("layers.{li}");
        let ln = b.constant(&format!("{p}.ln.weight"), TensorType::f32(vec![h]));
        let normed = rmsnorm(&b, x, ln, cfg.eps);

        let mix = if is_full {
            let kc_in = b.state_input(
                &format!("{p}.k_cache"),
                TensorType::f32(vec![1, cfg.n_kv_heads, l, cfg.head_dim]),
                StateRole::Recurrent,
            );
            let vc_in = b.state_input(
                &format!("{p}.v_cache"),
                TensorType::f32(vec![1, cfg.n_kv_heads, l, cfg.head_dim]),
                StateRole::Recurrent,
            );
            let (out, kc_out, vc_out) = qsa_attention_prefill_layer(
                &b,
                normed,
                &p,
                cfg,
                qcfg,
                l,
                cos,
                sin,
                block_cos,
                block_sin,
                causal_mask,
                block_eligible,
                tail_mask,
                kc_in,
                vc_in,
            );
            state.push((kc_in, kc_out));
            state.push((vc_in, vc_out));
            out
        } else {
            let key_dim = gk * ghd;
            let value_dim = gv * ghd;
            let conv_dim = 2 * key_dim + value_dim;
            let w_qkv = b.constant(
                &format!("{p}.qkv.weight"),
                TensorType::f32(vec![h, conv_dim]),
            );
            let w_gate = b.constant(
                &format!("{p}.gate.weight"),
                TensorType::f32(vec![h, value_dim]),
            );
            let w_conv = b.constant(
                &format!("{p}.conv.weight"),
                TensorType::f32(vec![gck, conv_dim]),
            );
            let w_beta = b.constant(&format!("{p}.beta.weight"), TensorType::f32(vec![h, gv]));
            let w_alpha = b.constant(&format!("{p}.alpha.weight"), TensorType::f32(vec![h, gv]));
            let dt_bias = b.constant(&format!("{p}.dt_bias"), TensorType::f32(vec![gv]));
            let ssm_a = b.constant(&format!("{p}.ssm_a"), TensorType::f32(vec![gv]));
            let norm_w = b.constant(&format!("{p}.ssm_norm.weight"), TensorType::f32(vec![ghd]));
            let w_out = b.constant(
                &format!("{p}.ssm_out.weight"),
                TensorType::f32(vec![value_dim, h]),
            );
            let s_in = b.state_input(
                &format!("{p}.ssm_state"),
                TensorType::f32(vec![1, gv, ghd, ghd]),
                StateRole::Recurrent,
            );
            let (out, cc_out, s_out) = qwen3next_gdn_prefill_block(
                &b,
                normed,
                w_qkv,
                w_gate,
                w_conv,
                w_beta,
                w_alpha,
                dt_bias,
                ssm_a,
                norm_w,
                w_out,
                s_in,
                tril_incl,
                tril_strict,
                gk,
                gv,
                ghd,
                gck,
                chunk,
                cfg.eps,
                GDN_HEAD_ORDER,
            );
            let cc_ph = b.state_input(
                &format!("{p}.conv_cache"),
                TensorType::f32(vec![1, gck - 1, conv_dim]),
                StateRole::Recurrent,
            );
            state.push((cc_ph, cc_out));
            state.push((s_in, s_out));
            out
        };
        x = b.binary(BinOp::Add, x, mix);

        let ln2 = b.constant(&format!("{p}.post_ln.weight"), TensorType::f32(vec![h]));
        let normed2 = rmsnorm(&b, x, ln2, cfg.eps);
        let ffn = deepseek2_dense_ffn(&b, normed2, h, h * 2, li);
        x = b.binary(BinOp::Add, x, ffn);
    }

    let final_ln = b.constant("model.norm.weight", TensorType::f32(vec![h]));
    let xn = rmsnorm(&b, x, final_ln, cfg.eps);
    let last = b.slice(xn, 1, l - 1, l);
    let lm_head = b.constant("lm_head.weight", TensorType::f32(vec![h, vocab]));
    let logits = linear(&b, last, lm_head, None);
    let g = b.finish_with_state(logits, &state);

    g.validate().expect("mixed GDN+QSA stack should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, vocab]);
}

/// [`trace_qwen38_prefill`] validates and produces the expected `[1,1,vocab]` shape over a small mixed
/// `[GDN,GDN,GDN,QSA]` schedule, with `chunk` a multiple of `index_compress_ratio` (the schedule of
/// `mixed_gdn_qsa_stack_validates_sc003`, through the top-level function).
#[test]
fn trace_qwen38_prefill_validates() {
    let mcfg = Qwen4ExpModelConfig {
        cfg: tiny_cfg(),
        qcfg: tiny_qcfg(4), // block_topk = 2, index_compress_ratio = 2
        gdn: Qwen4ExpGdnConfig {
            num_k_heads: 1,
            num_v_heads: 2,
            head_dim: 2,
            conv_k: 4,
        },
        layer_is_full: vec![false, false, false, true],
        vocab: 6,
        max_pos: 16,
        ffn_inter: 8,
        moe_n_experts: 3,
        moe_top_k: 2,
        moe_inter: 4,
        eps: 1e-5,
        chunk: 4, // multiple of index_compress_ratio=2 and conv_k=4
        hc_count: 2,
        hc_lowrank: 3,
        ple: None,
    };
    let g = trace_qwen38_prefill(&mcfg, 8);
    g.validate().expect("trace_qwen38_prefill should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, mcfg.vocab]);
    // 3 GDN layers (conv_cache + ssm_state) + 1 QSA layer (k_cache + v_cache) = 8 state pairs.
    assert_eq!(g.state.len(), 8);
}

/// The HF checkpoint pairs value head `j` with key head `j / n_rep` (`repeat_interleave`). With
/// `H_k = 2` and `H_v = 6`, grouped expansion broadcasts `[1, 2, 3, L, D]` while the GGUF tiled order
/// would broadcast `[1, 3, 2, L, D]`, so the expansion shape identifies the order in both traces.
#[test]
fn qwen38_gdn_expands_query_key_heads_in_hf_grouped_order() {
    let mcfg = Qwen4ExpModelConfig {
        cfg: tiny_cfg(),
        qcfg: tiny_qcfg(4),
        gdn: Qwen4ExpGdnConfig {
            num_k_heads: 2,
            num_v_heads: 6,
            head_dim: 2,
            conv_k: 4,
        },
        layer_is_full: vec![false, false, false, true],
        vocab: 6,
        max_pos: 16,
        ffn_inter: 8,
        moe_n_experts: 3,
        moe_top_k: 2,
        moe_inter: 4,
        eps: 1e-5,
        chunk: 4,
        hc_count: 2,
        hc_lowrank: 3,
        ple: None,
    };
    let broadcasts = |graph: &Graph, shape: [usize; 5]| {
        graph
            .eqns
            .iter()
            .filter(|eqn| matches!(&eqn.op, OpKind::Broadcast { shape: s } if s[..] == shape))
            .count()
    };
    // q and k of each of the three GDN layers.
    let prefill = trace_qwen38_prefill(&mcfg, 8);
    assert_eq!(broadcasts(&prefill, [1, 2, 3, 8, 2]), 6);
    assert_eq!(broadcasts(&prefill, [1, 3, 2, 8, 2]), 0);
    let decode = trace_qwen38_decode(&mcfg, 4);
    assert_eq!(broadcasts(&decode, [1, 2, 3, 1, 2]), 6);
    assert_eq!(broadcasts(&decode, [1, 3, 2, 1, 2]), 0);
}

/// A seq_len not a multiple of `qcfg.index_compress_ratio` must panic (QSA's block-pooling reshape,
/// `qsa_combined_mask`'s `nb * c == l` assert). `chunk` is also not a multiple of `seq_len=7`, showing
/// the panic comes from `index_compress_ratio` alone.
#[test]
#[should_panic(expected = "exact multiple of index_compress_ratio")]
fn trace_qwen38_prefill_rejects_non_multiple_of_compress_ratio() {
    let mcfg = Qwen4ExpModelConfig {
        cfg: tiny_cfg(),
        qcfg: tiny_qcfg(4), // index_compress_ratio = 2
        gdn: Qwen4ExpGdnConfig {
            num_k_heads: 1,
            num_v_heads: 2,
            head_dim: 2,
            conv_k: 4,
        },
        layer_is_full: vec![false, true],
        vocab: 6,
        max_pos: 16,
        ffn_inter: 8,
        moe_n_experts: 3,
        moe_top_k: 2,
        moe_inter: 4,
        eps: 1e-5,
        chunk: 4,
        hc_count: 2,
        hc_lowrank: 3,
        ple: None,
    };
    trace_qwen38_prefill(&mcfg, 7); // not a multiple of index_compress_ratio=2
}

/// `mcfg.chunk` is not a length constraint on `seq_len`; only `qcfg.index_compress_ratio` is
/// (spec 282 Out-of-scope revision). `l=6` is a multiple of `index_compress_ratio=2` but not of
/// `chunk=4`; it must validate and produce the expected shape, since
/// `qwen3next_gdn_prefill_block`'s internal zero-pad-and-truncate (FR-005) makes the ragged final GDN
/// tile transparent, as in the dense Qwen3-Next tracer.
#[test]
fn trace_qwen38_prefill_accepts_seq_len_not_multiple_of_chunk() {
    let mcfg = Qwen4ExpModelConfig {
        cfg: tiny_cfg(),
        qcfg: tiny_qcfg(4), // block_topk = 2, index_compress_ratio = 2
        gdn: Qwen4ExpGdnConfig {
            num_k_heads: 1,
            num_v_heads: 2,
            head_dim: 2,
            conv_k: 4,
        },
        layer_is_full: vec![false, false, false, true],
        vocab: 6,
        max_pos: 16,
        ffn_inter: 8,
        moe_n_experts: 3,
        moe_top_k: 2,
        moe_inter: 4,
        eps: 1e-5,
        chunk: 4, // l=6 is NOT a multiple of chunk=4, only of index_compress_ratio=2
        hc_count: 2,
        hc_lowrank: 3,
        ple: None,
    };
    let g = trace_qwen38_prefill(&mcfg, 6);
    g.validate().expect("trace_qwen38_prefill should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, mcfg.vocab]);
    assert_eq!(g.state.len(), 8);
}

/// [`trace_qwen38_decode`] validates and produces the expected `[1,1,vocab]` shape over the same small
/// mixed `[GDN,GDN,GDN,QSA]` schedule [`trace_qwen38_prefill_validates`] uses, at a fixed `cap`.
#[test]
fn trace_qwen38_decode_validates() {
    let mcfg = Qwen4ExpModelConfig {
        cfg: tiny_cfg(),
        qcfg: tiny_qcfg(4), // block_topk = 2, index_compress_ratio = 2
        gdn: Qwen4ExpGdnConfig {
            num_k_heads: 1,
            num_v_heads: 2,
            head_dim: 2,
            conv_k: 4,
        },
        layer_is_full: vec![false, false, false, true],
        vocab: 6,
        max_pos: 16,
        ffn_inter: 8,
        moe_n_experts: 3,
        moe_top_k: 2,
        moe_inter: 4,
        eps: 1e-5,
        chunk: 4,
        hc_count: 2,
        hc_lowrank: 3,
        ple: None,
    };
    let g = trace_qwen38_decode(&mcfg, 8); // cap=8, multiple of index_compress_ratio=2
    g.validate().expect("trace_qwen38_decode should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, mcfg.vocab]);
    // 3 GDN layers (conv_cache + ssm_state = 2 each) + 1 QSA layer (k_cache + v_cache + idx_k_cache =
    // 3) = 9 state pairs, one more than trace_qwen38_prefill's 8 (the QSA layer's indexer raw-key
    // history cache, see `qsa_attention_decode_layer`).
    assert_eq!(g.state.len(), 9);
}

/// A `cap` not a multiple of `index_compress_ratio` must panic; the decode counterpart of
/// `trace_qwen38_prefill_accepts_seq_len_not_multiple_of_chunk` (decode has no `chunk` dependency).
#[test]
#[should_panic(expected = "exact multiple of index_compress_ratio")]
fn trace_qwen38_decode_rejects_non_multiple_of_compress_ratio() {
    let mcfg = Qwen4ExpModelConfig {
        cfg: tiny_cfg(),
        qcfg: tiny_qcfg(4), // index_compress_ratio = 2
        gdn: Qwen4ExpGdnConfig {
            num_k_heads: 1,
            num_v_heads: 2,
            head_dim: 2,
            conv_k: 4,
        },
        layer_is_full: vec![false, true],
        vocab: 6,
        max_pos: 16,
        ffn_inter: 8,
        moe_n_experts: 3,
        moe_top_k: 2,
        moe_inter: 4,
        eps: 1e-5,
        chunk: 4,
        hc_count: 2,
        hc_lowrank: 3,
        ple: None,
    };
    trace_qwen38_decode(&mcfg, 7); // not a multiple of index_compress_ratio=2
}

mod cpu_oracle {
    use super::*;

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Real-magnitude synthetic weights: RMSNorm gammas near `1.0`, projections near `0.0`, as
    /// `crate::deepseek32`'s `cpu_oracle::weight` (never poot-vs-poot unit-scale values).
    fn weight(name: &str, n: usize, is_norm_gamma: bool) -> Vec<f32> {
        let raw = fill(n, seed_of(name));
        if is_norm_gamma {
            raw.iter().map(|v| 1.0 + v * 0.05).collect()
        } else {
            raw.iter().map(|v| v * 0.1).collect()
        }
    }

    struct LayerWeights {
        ln: Vec<f32>,
        wiq: Vec<f32>,
        q_ln: Vec<f32>,
        wik: Vec<f32>,
        k_ln: Vec<f32>,
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    }

    fn layer_weights(cfg: &Qwen4ExpConfig, qcfg: &QsaConfig, li: usize) -> LayerWeights {
        let h = cfg.hidden;
        let (hi, di) = (qcfg.index_n_heads, qcfg.index_head_dim);
        let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let p = |s: &str| format!("layers.{li}.{s}");
        LayerWeights {
            ln: weight(&p("ln"), h, true),
            wiq: weight(&p("indexer.wq"), h * hi * di, false),
            q_ln: weight(&p("indexer.q_ln"), di, true),
            wik: weight(&p("indexer.wk"), h * di, false),
            k_ln: weight(&p("indexer.k_ln"), di, true),
            wq: weight(&p("wq"), h * nh * 2 * hd, false),
            wk: weight(&p("wk"), h * nkv * hd, false),
            wv: weight(&p("wv"), h * nkv * hd, false),
            wo: weight(&p("wo"), nh * hd * h, false),
            q_norm: weight(&p("q_norm"), hd, true),
            k_norm: weight(&p("k_norm"), hd, true),
        }
    }

    fn rmsnorm_ref(x: &[f32], w: &[f32], n: usize, eps: f32) -> Vec<f32> {
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let den = (ms + eps).sqrt();
        (0..n).map(|i| x[i] / den * w[i]).collect()
    }

    fn linear_ref(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += x[i] * w[i * out_dim + o];
            }
            y[o] = acc;
        }
        y
    }

    fn sigmoid_ref(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    /// Independent reference for the half-split ("rotate-half") partial RoPE, matching
    /// `poot_graph_ir::ops::rope_prefill`'s `rope_partial` core; re-derived here (not imported from
    /// `crate::deepseek32`'s test helper) so this test does not depend on another module's test code.
    fn rope_half_split_ref(row: &[f32], cos_row: &[f32], sin_row: &[f32], rot: usize) -> Vec<f32> {
        let half = rot / 2;
        let mut out = row.to_vec();
        for i in 0..half {
            let (x1, x2) = (row[i], row[half + i]);
            out[i] = x1 * cos_row[i] - x2 * sin_row[i];
            out[half + i] = x2 * cos_row[half + i] + x1 * sin_row[half + i];
        }
        out
    }

    /// Independent QSA reference: computes indexer scores and picks the top blocks by a plain sort (not
    /// the graph's pairwise-`Ge` rank), and runs softmax attention restricted to the selected key set
    /// (not a full-width masked softmax), structured differently from `qsa_combined_mask`/
    /// `qwen3next_gated_attention_prefill`'s additive-mask-then-full-softmax, like
    /// `crate::deepseek32::dsa_attend_ref`.
    #[allow(clippy::too_many_arguments)]
    fn qsa_layer_ref(
        cfg: &Qwen4ExpConfig,
        qcfg: &QsaConfig,
        w: &LayerWeights,
        x: &[Vec<f32>], // [L][H], already normed
        cos: &[f32],
        sin: &[f32], // [L][rot], main table
        block_cos: &[f32],
        block_sin: &[f32], // [NB][rot], block table
    ) -> Vec<Vec<f32>> {
        let l = x.len();
        let (h, rot) = (cfg.hidden, cfg.rotary_dim);
        let (hi, di, c) = (
            qcfg.index_n_heads,
            qcfg.index_head_dim,
            qcfg.index_compress_ratio,
        );
        let nb = l / c;
        let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let n_rep = nh / nkv;

        // Indexer query per position per head, RoPE'd at its own position.
        let q_idx: Vec<Vec<Vec<f32>>> = (0..l)
            .map(|t| {
                let raw = linear_ref(&x[t], &w.wiq, h, hi * di);
                (0..hi)
                    .map(|hh| {
                        let head = &raw[hh * di..(hh + 1) * di];
                        let normed = rmsnorm_ref(head, &w.q_ln, di, cfg.eps);
                        rope_half_split_ref(
                            &normed,
                            &cos[t * rot..(t + 1) * rot],
                            &sin[t * rot..(t + 1) * rot],
                            rot,
                        )
                    })
                    .collect()
            })
            .collect();

        // Raw (unrotated, unnormalized) indexer keys, then block-pooled + normed + RoPE'd.
        let raw_k: Vec<Vec<f32>> = (0..l).map(|t| linear_ref(&x[t], &w.wik, h, di)).collect();
        let block_key: Vec<Vec<f32>> = (0..nb)
            .map(|bi| {
                let mut pooled = vec![0.0f32; di];
                for row in &raw_k[bi * c..bi * c + c] {
                    for d in 0..di {
                        pooled[d] += row[d];
                    }
                }
                for v in pooled.iter_mut() {
                    *v /= c as f32;
                }
                let normed = rmsnorm_ref(&pooled, &w.k_ln, di, cfg.eps);
                rope_half_split_ref(
                    &normed,
                    &block_cos[bi * rot..(bi + 1) * rot],
                    &block_sin[bi * rot..(bi + 1) * rot],
                    rot,
                )
            })
            .collect();

        // Main attention q/gate/k/v, RoPE'd where applicable.
        let scale = 1.0 / (hd as f32).sqrt();
        let mut q_main = vec![vec![vec![0.0f32; hd]; nh]; l];
        let mut gate_main = vec![vec![vec![0.0f32; hd]; nh]; l];
        for t in 0..l {
            let raw = linear_ref(&x[t], &w.wq, h, nh * 2 * hd);
            for hh in 0..nh {
                let base = hh * 2 * hd;
                let q_raw = &raw[base..base + hd];
                let g_raw = &raw[base + hd..base + 2 * hd];
                let q_normed = rmsnorm_ref(q_raw, &w.q_norm, hd, cfg.eps);
                q_main[t][hh] = rope_half_split_ref(
                    &q_normed,
                    &cos[t * rot..(t + 1) * rot],
                    &sin[t * rot..(t + 1) * rot],
                    rot,
                );
                gate_main[t][hh] = g_raw.iter().map(|&v| sigmoid_ref(v)).collect();
            }
        }
        let mut k_main = vec![vec![vec![0.0f32; hd]; nkv]; l];
        let mut v_main = vec![vec![vec![0.0f32; hd]; nkv]; l];
        for t in 0..l {
            let kraw = linear_ref(&x[t], &w.wk, h, nkv * hd);
            let vraw = linear_ref(&x[t], &w.wv, h, nkv * hd);
            for kh in 0..nkv {
                let k_normed = rmsnorm_ref(&kraw[kh * hd..(kh + 1) * hd], &w.k_norm, hd, cfg.eps);
                k_main[t][kh] = rope_half_split_ref(
                    &k_normed,
                    &cos[t * rot..(t + 1) * rot],
                    &sin[t * rot..(t + 1) * rot],
                    rot,
                );
                v_main[t][kh] = vraw[kh * hd..(kh + 1) * hd].to_vec();
            }
        }

        (0..l)
            .map(|t| {
                // Indexer: score every eligible (complete) block, top-k by plain sort, plus the
                // always-visible tail (the independent counterpart of FR-002/FR-003/FR-004).
                let num_complete = (t + 1) / c;
                let mut block_scores: Vec<(usize, f32)> = (0..num_complete)
                    .map(|bi| {
                        let mut score = 0.0f32;
                        for q_idx_head in &q_idx[t] {
                            let mut dot = 0.0f32;
                            for d in 0..di {
                                dot += q_idx_head[d] * block_key[bi][d];
                            }
                            score += dot.max(0.0);
                        }
                        (bi, score / (di as f32).sqrt())
                    })
                    .collect();
                block_scores.sort_by(|a, b2| b2.1.partial_cmp(&a.1).unwrap());
                let keep = qcfg.block_topk().min(num_complete);
                let mut selected: Vec<usize> = block_scores[..keep]
                    .iter()
                    .flat_map(|&(bi, _)| bi * c..bi * c + c)
                    .collect();
                let tail_start = num_complete * c;
                if tail_start <= t {
                    selected.extend(tail_start..=t);
                }
                selected.sort_unstable();

                // Restricted softmax attention over `selected`, per head.
                let mut out_row = vec![0.0f32; nh * hd];
                for hh in 0..nh {
                    let kv = hh / n_rep;
                    let mut scores = Vec::with_capacity(selected.len());
                    for &j in &selected {
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += q_main[t][hh][d] * k_main[j][kv][d];
                        }
                        scores.push(dot * scale);
                    }
                    let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
                    let denom: f32 = exp.iter().sum();
                    for d in 0..hd {
                        let mut acc = 0.0f32;
                        for (si, &j) in selected.iter().enumerate() {
                            acc += (exp[si] / denom) * v_main[j][kv][d];
                        }
                        out_row[hh * hd + d] = acc * gate_main[t][hh][d];
                    }
                }
                linear_ref(&out_row, &w.wo, nh * hd, h)
            })
            .collect()
    }

    fn qsa_probe_ref(
        cfg: &Qwen4ExpConfig,
        qcfg: &QsaConfig,
        layers: usize,
        x0: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let l = x0.len();
        let nb = l / qcfg.index_compress_ratio;
        let (cos, sin) = qwen38_rope_tables(l, cfg.rotary_dim, 10_000.0);
        let (bcos, bsin) =
            qwen38_block_rope_tables(nb, qcfg.index_compress_ratio, cfg.rotary_dim, 10_000.0);
        let final_ln = weight("final.ln.weight", cfg.hidden, true);

        let mut x = x0.to_vec();
        for li in 0..layers {
            let w = layer_weights(cfg, qcfg, li);
            let normed: Vec<Vec<f32>> = x
                .iter()
                .map(|row| rmsnorm_ref(row, &w.ln, cfg.hidden, cfg.eps))
                .collect();
            let attn = qsa_layer_ref(cfg, qcfg, &w, &normed, &cos, &sin, &bcos, &bsin);
            for t in 0..l {
                for c in 0..cfg.hidden {
                    x[t][c] += attn[t][c];
                }
            }
        }
        x.iter()
            .map(|row| rmsnorm_ref(row, &final_ln, cfg.hidden, cfg.eps))
            .collect()
    }

    fn eval_probe(
        g: &Graph,
        x0: &[Vec<f32>],
        weights: &HashMap<String, Vec<f32>>,
        l: usize,
        h: usize,
        compress_ratio: usize,
    ) -> Vec<Vec<f32>> {
        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let t = match &meta.storage {
                Storage::State => poot_tensor::HostTensor::zeros(meta.aval.shape.clone()),
                Storage::Slot(Slot::Mask) => {
                    let name = meta.name.as_deref().expect("mask slot without a name");
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    poot_tensor::HostTensor::f32(meta.aval.shape.clone(), causal_mask_data(l))
                }
                Storage::Slot(Slot::Activation) => {
                    let name = meta
                        .name
                        .as_deref()
                        .expect("activation slot without a name");
                    let data = if name == "activation.qsa.block_eligible" {
                        let nb = meta.aval.shape[3];
                        block_eligible_mask_data(l, nb, compress_ratio)
                    } else if name == "activation.qsa.tail_mask" {
                        tail_mask_data(l, compress_ratio)
                    } else {
                        panic!("unexpected activation slot {name} in the prefill probe")
                    };
                    poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data)
                }
                Storage::Const => {
                    let name = meta.name.as_deref().expect("const without a name");
                    let data = if name == "probe.x0" {
                        x0.iter().flatten().copied().collect()
                    } else {
                        crate::model_fixture_data(weights, name)
                    };
                    poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data)
                }
                other => {
                    panic!("unexpected storage {other:?} in a stateful-but-fresh prefill probe")
                }
            };
            inputs.insert(id, t.into());
        }
        let out = poot_eval::eval(
            g,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("cpu eval")
        .output
        .into_host()
        .expect("dense output");
        let flat = out.as_f32().unwrap();
        (0..l).map(|t| flat[t * h..(t + 1) * h].to_vec()).collect()
    }

    fn all_weights(
        cfg: &Qwen4ExpConfig,
        qcfg: &QsaConfig,
        layers: usize,
    ) -> HashMap<String, Vec<f32>> {
        let h = cfg.hidden;
        let mut w = HashMap::new();
        for li in 0..layers {
            let lw = layer_weights(cfg, qcfg, li);
            let p = |s: &str| format!("layers.{li}.{s}");
            w.insert(p("ln.weight"), lw.ln);
            w.insert(p("indexer.wq.weight"), lw.wiq);
            w.insert(p("indexer.q_layernorm.weight"), lw.q_ln);
            w.insert(p("indexer.wk.weight"), lw.wik);
            w.insert(p("indexer.k_layernorm.weight"), lw.k_ln);
            w.insert(p("q_proj.weight"), lw.wq);
            w.insert(p("k_proj.weight"), lw.wk);
            w.insert(p("v_proj.weight"), lw.wv);
            w.insert(p("o_proj.weight"), lw.wo);
            w.insert(p("q_norm.weight"), lw.q_norm);
            w.insert(p("k_norm.weight"), lw.k_norm);
        }
        w.insert(
            "final.ln.weight".to_string(),
            weight("final.ln.weight", h, true),
        );
        w
    }

    fn rope_table_weights(
        cfg: &Qwen4ExpConfig,
        qcfg: &QsaConfig,
        l: usize,
    ) -> HashMap<String, Vec<f32>> {
        let nb = l / qcfg.index_compress_ratio;
        let (cos, sin) = qwen38_rope_tables(l, cfg.rotary_dim, 10_000.0);
        let (bcos, bsin) =
            qwen38_block_rope_tables(nb, qcfg.index_compress_ratio, cfg.rotary_dim, 10_000.0);
        let mut w = HashMap::new();
        w.insert("rope.cos".to_string(), cos);
        w.insert("rope.sin".to_string(), sin);
        w.insert("qsa.block_rope.cos".to_string(), bcos);
        w.insert("qsa.block_rope.sin".to_string(), bsin);
        w
    }

    fn synthetic_x0(l: usize, h: usize) -> Vec<Vec<f32>> {
        let raw = fill(l * h, seed_of("probe.x0"));
        (0..l)
            .map(|t| raw[t * h..(t + 1) * h].iter().map(|v| v * 0.5).collect())
            .collect()
    }

    /// SC-001: the graph tracer's `poot-eval` output matches the independent reference within `1e-4` at
    /// every query position (not just the last). `L=8`, `C=2`, `block_topk=2` covers the four required
    /// cases: a pure-tail query (t=0), an exact block-boundary query with zero tail (t=1, t=3), a
    /// fully-under-budget query (t=3: 2 complete blocks, budget 2, both kept), and a real over-budget
    /// drop (t=5: 3 complete blocks; t=7: 4 complete blocks; budget 2).
    #[test]
    fn qwen38_qsa_probe_matches_independent_reference() {
        let cfg = tiny_cfg();
        let qcfg = tiny_qcfg(4); // block_topk = 2
        let layers = 2;
        let l = 8;

        let g = trace_qwen38_qsa_probe(cfg, qcfg, layers, l);
        let x0 = synthetic_x0(l, cfg.hidden);
        let mut weights = all_weights(&cfg, &qcfg, layers);
        weights.extend(rope_table_weights(&cfg, &qcfg, l));

        let got = eval_probe(&g, &x0, &weights, l, cfg.hidden, qcfg.index_compress_ratio);
        let want = qsa_probe_ref(&cfg, &qcfg, layers, &x0);

        for t in 0..l {
            for c in 0..cfg.hidden {
                let (a, b) = (got[t][c], want[t][c]);
                assert!(
                    (a - b).abs() < 1e-4,
                    "position {t}, channel {c}: got {a}, want {b}"
                );
            }
        }
    }

    /// SC-002: a deliberate mutation breaks the differential test and reverting it passes again. The
    /// mutation runs the independent reference with `index_compress_ratio + 1` while the traced graph
    /// keeps the original, changing the reference's block partition, tail-window boundaries, and top-k
    /// competition (spec 282 FR-002/FR-003/FR-004), so a real divergence must appear. The same `got`
    /// (graph output, computed once) is compared against the mutated and original reference, isolating
    /// the mutation as the only variable.
    #[test]
    fn mutation_breaks_then_revert_restores_qwen38_qsa_probe_match() {
        let cfg = tiny_cfg();
        let qcfg = tiny_qcfg(4);
        let layers = 2;
        let l = 8;

        let g = trace_qwen38_qsa_probe(cfg, qcfg, layers, l);
        let x0 = synthetic_x0(l, cfg.hidden);
        let mut weights = all_weights(&cfg, &qcfg, layers);
        weights.extend(rope_table_weights(&cfg, &qcfg, l));
        let got = eval_probe(&g, &x0, &weights, l, cfg.hidden, qcfg.index_compress_ratio);

        // Mutated reference: index_compress_ratio + 1, so the tail/block partition disagrees with the
        // graph's compress_ratio (reachable by mutating the CONFIG the reference runs with, not the
        // graph, so `qsa_layer_ref`'s body is not duplicated).
        let mutated_qcfg = QsaConfig {
            index_compress_ratio: qcfg.index_compress_ratio + 1,
            ..qcfg
        };
        let mutated_want = qsa_probe_ref(&cfg, &mutated_qcfg, layers, &x0);
        let mut mismatched = false;
        'outer: for t in 0..l {
            for c in 0..cfg.hidden {
                if (got[t][c] - mutated_want[t][c]).abs() >= 1e-4 {
                    mismatched = true;
                    break 'outer;
                }
            }
        }
        assert!(
            mismatched,
            "mutated compress_ratio should break the match (SC-002)"
        );

        // Revert: the original reference must match again, as in the primary test.
        let restored_want = qsa_probe_ref(&cfg, &qcfg, layers, &x0);
        for t in 0..l {
            for c in 0..cfg.hidden {
                assert!(
                    (got[t][c] - restored_want[t][c]).abs() < 1e-4,
                    "reverted comparison should match again at position {t}, channel {c}"
                );
            }
        }
    }

    /// Plain causal decode-step mask row `[cap]`: `0.0` for `t <= pos`, [`QSA_MASK_NEG`] for `t > pos`.
    /// A test-local copy of `poot_llm::graphs::decode_mask_row`'s convention (this crate cannot depend
    /// on `poot-llm`), used to bind `Slot::Mask` in the decode-step tests below.
    fn decode_causal_mask_row(cap: usize, pos: usize) -> Vec<f32> {
        (0..cap)
            .map(|t| if t <= pos { 0.0 } else { QSA_MASK_NEG })
            .collect()
    }

    /// Bind and evaluate one decode step of a graph built by [`trace_qwen38_qsa_probe_decode`] or
    /// [`trace_qwen38_decode`]: `Slot::Pos`/`Slot::Mask` from `pos`/`cap`, the
    /// `qsa.block_positions` step input synthesized from `compress_ratio` (the
    /// `poot_llm::generate::bind_decode` equivalent), `probe.x0` from `x0`, carried `state` (empty on
    /// the first call, meaning zero-fill, like `eval_probe`'s `Storage::State` arm), and every other
    /// named `Const` from `weights` (`causal.iota` is an in-graph `iota`, card 550a, so no arm binds
    /// it). `token` is `None` for the QSA-only probe (no `Slot::Token`/embed table; `x0` is fed
    /// via `probe.x0`) and `Some(id)` for the whole-model trace. Returns `(output_row, new_state)`.
    #[allow(clippy::too_many_arguments)]
    fn eval_decode_step(
        g: &Graph,
        token: Option<u32>,
        pos: usize,
        cap: usize,
        compress_ratio: usize,
        x0: Option<&[f32]>,
        weights: &HashMap<String, Vec<f32>>,
        state: &[poot_tensor::HostTensor],
    ) -> (Vec<f32>, Vec<poot_tensor::HostTensor>) {
        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let t = match &meta.storage {
                Storage::Slot(Slot::Token) => poot_tensor::HostTensor::i32(
                    vec![],
                    vec![token.expect("Slot::Token present") as i32],
                ),
                Storage::Slot(Slot::Pos) => poot_tensor::HostTensor::i32(vec![], vec![pos as i32]),
                Storage::Slot(Slot::SeqLen) => {
                    poot_tensor::HostTensor::i32(vec![], vec![(pos + 1) as i32])
                }
                Storage::Slot(Slot::Mask) => poot_tensor::HostTensor::f32(
                    meta.aval.shape.clone(),
                    decode_causal_mask_row(cap, pos),
                ),
                Storage::Slot(Slot::Activation) => {
                    let name = meta
                        .name
                        .as_deref()
                        .expect("activation slot without a name");
                    assert_eq!(
                        name, "activation.qsa.block_positions",
                        "unexpected activation slot {name}"
                    );
                    let nb = meta.aval.shape[0];
                    poot_tensor::HostTensor::f32(
                        meta.aval.shape.clone(),
                        (0..nb).map(|bi| (bi * compress_ratio) as f32).collect(),
                    )
                }
                Storage::Slot(other) => panic!("unexpected slot {other:?} in decode probe"),
                Storage::State => continue, // bound below, in g.state order
                Storage::Const => {
                    let name = meta.name.as_deref().expect("const without a name");
                    let data = if name == "probe.x0" {
                        x0.expect("probe.x0 present").to_vec()
                    } else {
                        crate::model_fixture_data(weights, name)
                    };
                    poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data)
                }
                other => panic!("unexpected storage {other:?} in decode probe"),
            };
            inputs.insert(id, t.into());
        }
        for (ci, &(si, _)) in g.state.iter().enumerate() {
            let t = state
                .get(ci)
                .cloned()
                .unwrap_or_else(|| poot_tensor::HostTensor::zeros(g.aval(si).shape.clone()));
            inputs.insert(si, t.into());
        }
        let step = poot_eval::eval(
            g,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("cpu eval");
        let out = step.output.into_host().expect("dense output");
        let new_state: Vec<poot_tensor::HostTensor> = step
            .state
            .into_iter()
            .map(|v| v.into_host().expect("dense state"))
            .collect();
        (out.as_f32().unwrap().to_vec(), new_state)
    }

    /// SC-004: [`trace_qwen38_qsa_probe_decode`]'s per-step output matches [`qsa_probe_ref`] at every
    /// position of a `cap=8` trajectory within `1e-4`, the decode counterpart of
    /// `qwen38_qsa_probe_matches_independent_reference`, exercising the dynamic `pos`-based
    /// block-eligibility/tail derivation, the recompute-from-raw-cache pooling, and `attention_masked`
    /// with a `Slot::Mask`-derived causal term against the same reference. Sequential: `state` is
    /// threaded from each step into the next, starting from an empty (zero-filled) cache.
    #[test]
    fn qwen38_qsa_probe_decode_matches_independent_reference() {
        let cfg = tiny_cfg();
        let qcfg = tiny_qcfg(4); // block_topk = 2
        let layers = 2;
        let cap = 8;

        let g = trace_qwen38_qsa_probe_decode(cfg, qcfg, layers, cap);
        let x0 = synthetic_x0(cap, cfg.hidden);
        let mut weights = all_weights(&cfg, &qcfg, layers);
        weights.extend(rope_table_weights(&cfg, &qcfg, cap));
        let want = qsa_probe_ref(&cfg, &qcfg, layers, &x0);

        let mut state: Vec<poot_tensor::HostTensor> = Vec::new();
        for pos in 0..cap {
            let (got_row, new_state) = eval_decode_step(
                &g,
                None,
                pos,
                cap,
                qcfg.index_compress_ratio,
                Some(&x0[pos]),
                &weights,
                &state,
            );
            state = new_state;
            for ci in 0..cfg.hidden {
                let (a, b) = (got_row[ci], want[pos][ci]);
                assert!(
                    (a - b).abs() < 1e-4,
                    "position {pos}, channel {ci}: got {a}, want {b}"
                );
            }
        }
    }

    fn tiny_gdn_cfg() -> Qwen4ExpGdnConfig {
        Qwen4ExpGdnConfig {
            num_k_heads: 1,
            num_v_heads: 2,
            head_dim: 2,
            conv_k: 4,
        }
    }

    /// A larger-magnitude, distinctly-negative synthetic weight (GDN `ssm_a` is always negative, so
    /// `exp(alpha * ssm_a)` stays in `(0,1)`; see `qwen3next_gdn_block`), as `weight`'s real-magnitude
    /// discipline shifted negative.
    fn neg_weight(name: &str, n: usize) -> Vec<f32> {
        fill(n, seed_of(name))
            .iter()
            .map(|v| -0.3 - v.abs() * 0.2)
            .collect()
    }

    /// Every named `Const` [`trace_qwen38_prefill`]/[`trace_qwen38_decode`] can declare for `mcfg`'s
    /// mixed GDN/QSA schedule: real-magnitude synthetic weights, `model.embed_tokens.weight` sized to
    /// `mcfg.vocab` rows (token ids index directly into it, one row per desired `x0`). Excludes dynamic
    /// (seq/`pos`-dependent) consts, which `eval_decode_step` and the prefill-side inline binder in
    /// `qwen38_decode_loop_matches_prefill` synthesize per call, as `Runner::bind`/`bind_decode` do.
    fn whole_model_weights(mcfg: &Qwen4ExpModelConfig) -> HashMap<String, Vec<f32>> {
        let h = mcfg.cfg.hidden;
        let (hi, di) = (mcfg.qcfg.index_n_heads, mcfg.qcfg.index_head_dim);
        let (nh, nkv, hd) = (mcfg.cfg.n_heads, mcfg.cfg.n_kv_heads, mcfg.cfg.head_dim);
        let (gk, gv, ghd, gck) = (
            mcfg.gdn.num_k_heads,
            mcfg.gdn.num_v_heads,
            mcfg.gdn.head_dim,
            mcfg.gdn.conv_k,
        );
        let key_dim = gk * ghd;
        let value_dim = gv * ghd;
        let conv_dim = 2 * key_dim + value_dim;

        let (hc_count, hc_lowrank) = (mcfg.hc_count, mcfg.hc_lowrank);
        let hc_h = hc_count * h;

        let mut w = HashMap::new();
        w.insert(
            "model.embed_tokens.weight".to_string(),
            weight("model.embed_tokens.weight", mcfg.vocab * h, false),
        );
        w.insert(
            "lm_head.weight".to_string(),
            weight("lm_head.weight", h * mcfg.vocab, false),
        );

        // Model-level `hyper_connection_mixer` (`use_combine=False`, no `block_inject_weight`),
        // collapsing the final widened stream to one stream before `lm_head`.
        let insert_gated_residual =
            |w: &mut HashMap<String, Vec<f32>>, prefix: &str, with_inject: bool| {
                w.insert(
                    format!("{prefix}.hc_norm.weight"),
                    weight(&format!("{prefix}.hc_norm"), hc_h, true),
                );
                w.insert(
                    format!("{prefix}.input_mix_weight_down.weight"),
                    weight(&format!("{prefix}.mix_down"), hc_h * hc_lowrank, false),
                );
                w.insert(
                    format!("{prefix}.input_mix_weight_up.weight"),
                    weight(&format!("{prefix}.mix_up"), hc_lowrank * hc_h, false),
                );
                if with_inject {
                    w.insert(
                        format!("{prefix}.block_inject_weight.weight"),
                        weight(&format!("{prefix}.inject"), hc_h * hc_count, false),
                    );
                }
            };
        insert_gated_residual(&mut w, "hyper_connection_mixer", false);

        for (li, &is_full) in mcfg.layer_is_full.iter().enumerate() {
            let p = |s: &str| format!("layers.{li}.{s}");
            insert_gated_residual(&mut w, &p("attn_hyper_connection"), true);
            insert_gated_residual(&mut w, &p("mlp_hyper_connection"), true);
            if is_full {
                w.insert(
                    p("indexer.wq.weight"),
                    weight(&p("indexer.wq"), h * hi * di, false),
                );
                w.insert(
                    p("indexer.q_layernorm.weight"),
                    weight(&p("indexer.q_ln"), di, true),
                );
                w.insert(
                    p("indexer.wk.weight"),
                    weight(&p("indexer.wk"), h * di, false),
                );
                w.insert(
                    p("indexer.k_layernorm.weight"),
                    weight(&p("indexer.k_ln"), di, true),
                );
                w.insert(p("q_proj.weight"), weight(&p("wq"), h * nh * 2 * hd, false));
                w.insert(p("k_proj.weight"), weight(&p("wk"), h * nkv * hd, false));
                w.insert(p("v_proj.weight"), weight(&p("wv"), h * nkv * hd, false));
                w.insert(p("o_proj.weight"), weight(&p("wo"), nh * hd * h, false));
                w.insert(p("q_norm.weight"), weight(&p("q_norm"), hd, true));
                w.insert(p("k_norm.weight"), weight(&p("k_norm"), hd, true));
            } else {
                w.insert(
                    p("linear_attn.in_proj_qkv.weight"),
                    weight(&p("qkv"), h * conv_dim, false),
                );
                w.insert(
                    p("linear_attn.in_proj_z.weight"),
                    weight(&p("z"), h * value_dim, false),
                );
                w.insert(
                    p("linear_attn.conv1d.weight"),
                    weight(&p("conv"), gck * conv_dim, false),
                );
                w.insert(
                    p("linear_attn.in_proj_b.weight"),
                    weight(&p("beta"), h * gv, false),
                );
                w.insert(
                    p("linear_attn.in_proj_a.weight"),
                    weight(&p("alpha"), h * gv, false),
                );
                w.insert(p("linear_attn.dt_bias"), weight(&p("dt_bias"), gv, false));
                w.insert(p("linear_attn.ssm_a"), neg_weight(&p("ssm_a"), gv));
                w.insert(
                    p("linear_attn.norm.weight"),
                    weight(&p("ssm_norm"), ghd, true),
                );
                w.insert(
                    p("linear_attn.out_proj.weight"),
                    weight(&p("ssm_out"), value_dim * h, false),
                );
            }
            w.insert(
                p("mlp.shared_expert.gate_proj.weight"),
                weight(&p("ffn_gate"), h * mcfg.ffn_inter, false),
            );
            w.insert(
                p("mlp.shared_expert.up_proj.weight"),
                weight(&p("ffn_up"), h * mcfg.ffn_inter, false),
            );
            w.insert(
                p("mlp.shared_expert.down_proj.weight"),
                weight(&p("ffn_down"), mcfg.ffn_inter * h, false),
            );
            w.insert(
                p("mlp.shared_expert_gate.weight"),
                weight(&p("ffn_gate_scalar"), h, false),
            );
            w.insert(
                p("mlp.gate.weight"),
                weight(&p("moe_gate"), h * mcfg.moe_n_experts, false),
            );
            w.insert(
                p("mlp.experts.gate_up_proj"),
                weight(
                    &p("moe_gate_up"),
                    mcfg.moe_n_experts * h * 2 * mcfg.moe_inter,
                    false,
                ),
            );
            w.insert(
                p("mlp.experts.down_proj"),
                weight(
                    &p("moe_down"),
                    mcfg.moe_n_experts * mcfg.moe_inter * h,
                    false,
                ),
            );
        }
        w
    }

    /// SC-005: [`trace_qwen38_decode`]'s per-step output, run sequentially over a `cap=L` trajectory from
    /// an empty cache, matches [`trace_qwen38_prefill`]'s single `L`-token call at the last position
    /// (as `crate::deepseek4`'s `deepseek4_hybrid_stack_prefill_matches_decode_loop_*`). Both share
    /// `whole_model_weights`, so this isolates the decode-only code
    /// (`qsa_decode_block_eligible_mask`/`qsa_decode_tail_block_mask`/`qsa_combined_mask_decode`/the
    /// raw-key-cache recompute) rather than re-verifying QSA's shared block/tail math, which the
    /// independent-reference tests cover.
    #[test]
    fn qwen38_decode_loop_matches_prefill() {
        let mcfg = Qwen4ExpModelConfig {
            cfg: tiny_cfg(),
            qcfg: tiny_qcfg(4), // block_topk = 2, index_compress_ratio = 2
            gdn: tiny_gdn_cfg(),
            layer_is_full: vec![false, false, false, true],
            vocab: 8, // == l, one embed row per token id used below
            max_pos: 16,
            ffn_inter: 6,
            moe_n_experts: 3,
            moe_top_k: 2,
            moe_inter: 4,
            eps: 1e-5,
            chunk: 4, // multiple of index_compress_ratio=2 and conv_k=4; prefill-only
            hc_count: 2,
            hc_lowrank: 3,
            ple: None,
        };
        let l = 8; // multiple of chunk=4 and index_compress_ratio=2
        let tokens: Vec<u32> = (0..l as u32).collect(); // one distinct row per position

        let mut weights = whole_model_weights(&mcfg);
        let max_blocks = mcfg.max_pos / mcfg.qcfg.index_compress_ratio;
        let (cos, sin) = qwen38_rope_tables(mcfg.max_pos, mcfg.cfg.rotary_dim, 10_000.0);
        let (bcos, bsin) = qwen38_block_rope_tables(
            max_blocks,
            mcfg.qcfg.index_compress_ratio,
            mcfg.cfg.rotary_dim,
            10_000.0,
        );
        weights.insert("rope.cos".to_string(), cos);
        weights.insert("rope.sin".to_string(), sin);
        weights.insert("qsa.block_rope.cos".to_string(), bcos);
        weights.insert("qsa.block_rope.sin".to_string(), bsin);

        // --- prefill: one call over the full `l`-token sequence. ---
        let pg = trace_qwen38_prefill(&mcfg, l);
        let mut pinputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &pg.inputs {
            let meta = pg.meta(id);
            let t = match &meta.storage {
                Storage::Slot(Slot::Token) => poot_tensor::HostTensor::i32(
                    vec![l],
                    tokens.iter().map(|&t| t as i32).collect(),
                ),
                Storage::Slot(Slot::Mask) => {
                    let name = meta.name.as_deref().expect("mask slot without a name");
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    poot_tensor::HostTensor::f32(meta.aval.shape.clone(), causal_mask_data(l))
                }
                Storage::Slot(Slot::Activation) => {
                    let name = meta
                        .name
                        .as_deref()
                        .expect("activation slot without a name");
                    let data = if name == "activation.qsa.block_eligible" {
                        block_eligible_mask_data(
                            l,
                            meta.aval.shape[3],
                            mcfg.qcfg.index_compress_ratio,
                        )
                    } else if name == "activation.qsa.tail_mask" {
                        tail_mask_data(l, mcfg.qcfg.index_compress_ratio)
                    } else {
                        panic!("unexpected activation slot {name} in whole-model prefill")
                    };
                    poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data)
                }
                Storage::State => poot_tensor::HostTensor::zeros(meta.aval.shape.clone()),
                Storage::Const => {
                    let name = meta.name.as_deref().expect("const without a name");
                    let data = crate::model_fixture_data(&weights, name);
                    poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data)
                }
                other => panic!("unexpected storage {other:?} in whole-model prefill"),
            };
            pinputs.insert(id, t.into());
        }
        let prefill_out = poot_eval::eval(
            &pg,
            &pinputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("cpu eval")
        .output
        .into_host()
        .expect("dense output");

        // --- decode: `l` sequential single-token steps, cap == l. ---
        let dg = trace_qwen38_decode(&mcfg, l);
        let mut state: Vec<poot_tensor::HostTensor> = Vec::new();
        let mut decode_out = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            let (out_row, new_state) = eval_decode_step(
                &dg,
                Some(tok),
                pos,
                l,
                mcfg.qcfg.index_compress_ratio,
                None,
                &weights,
                &state,
            );
            state = new_state;
            decode_out = out_row;
        }

        assert_eq!(prefill_out.as_f32().unwrap().len(), decode_out.len());
        // decode loop (actual) vs prefill (expected), per vocab index
        poot_test_util::assert_close(&decode_out, prefill_out.as_f32().unwrap(), 1e-4);
    }
}

/// Independent-reference differential test for [`qwen38_moe_ffn`] (spec 282): a small standalone graph
/// exercising only the FFN composition, checked against a hand-written pure-Rust reference of
/// softmax-topk routing, per-expert SwiGLU, and shared-expert sigmoid gating (not a second graph-IR
/// construction).
mod moe_ffn_oracle {
    use super::*;

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 0.3
            })
            .collect()
    }

    /// A standalone graph: `x = probe.x` -> [`qwen38_moe_ffn`] -> output, `[1, L, H]`. No embed,
    /// residual, or `lm_head`.
    #[allow(clippy::too_many_arguments)]
    fn trace_moe_probe(
        h: usize,
        n_experts: usize,
        top_k: usize,
        inter: usize,
        shexp_inter: usize,
        l: usize,
    ) -> Graph {
        let b = Builder::new();
        let x = b.constant("probe.x", TensorType::f32(vec![1, l, h]));
        let out = qwen38_moe_ffn(&b, x, "probe", h, n_experts, top_k, inter, shexp_inter);
        b.finish(out)
    }

    struct MoeWeights {
        router: Vec<f32>, // [H, E]
        w_in: Vec<f32>,   // [E, H, 2I]
        w_out: Vec<f32>,  // [E, I, H]
        sg: Vec<f32>,     // [H, Ish]
        su: Vec<f32>,     // [H, Ish]
        sd: Vec<f32>,     // [Ish, H]
        sgi: Vec<f32>,    // [H, 1]
    }

    fn synth_weights(h: usize, e: usize, inter: usize, shexp_inter: usize) -> MoeWeights {
        MoeWeights {
            router: fill(h * e, seed_of("router")),
            w_in: fill(e * h * 2 * inter, seed_of("w_in")),
            w_out: fill(e * inter * h, seed_of("w_out")),
            sg: fill(h * shexp_inter, seed_of("sg")),
            su: fill(h * shexp_inter, seed_of("su")),
            sd: fill(shexp_inter * h, seed_of("sd")),
            sgi: fill(h, seed_of("sgi")),
        }
    }

    fn eval_probe(g: &Graph, x: &[Vec<f32>], w: &MoeWeights, l: usize, h: usize) -> Vec<Vec<f32>> {
        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let name = meta.name.as_deref().expect("const without a name");
            let data: Vec<f32> = if name == "probe.x" {
                x.iter().flatten().copied().collect()
            } else if name == "probe.mlp.gate.weight" {
                w.router.clone()
            } else if name == "probe.mlp.experts.gate_up_proj" {
                w.w_in.clone()
            } else if name == "probe.mlp.experts.down_proj" {
                w.w_out.clone()
            } else if name == "probe.mlp.shared_expert.gate_proj.weight" {
                w.sg.clone()
            } else if name == "probe.mlp.shared_expert.up_proj.weight" {
                w.su.clone()
            } else if name == "probe.mlp.shared_expert.down_proj.weight" {
                w.sd.clone()
            } else if name == "probe.mlp.shared_expert_gate.weight" {
                w.sgi.clone()
            } else {
                panic!("unexpected constant {name}")
            };
            inputs.insert(
                id,
                poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data).into(),
            );
        }
        let out = poot_eval::eval(
            g,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("cpu eval")
        .output
        .into_host()
        .expect("dense output");
        (0..l)
            .map(|t| out.as_f32().unwrap()[t * h..(t + 1) * h].to_vec())
            .collect()
    }

    fn silu(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    fn sigmoid_ref(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    /// Independent reference: softmax over all experts, top-k select, renormalize, per-expert SwiGLU
    /// weighted sum, plus sigmoid-gated shared-expert SwiGLU (`Qwen4ExpTextSparseMoeBlock.forward`; see
    /// [`qwen38_moe_ffn`]).
    #[allow(clippy::too_many_arguments)]
    fn moe_ffn_ref(
        x: &[f32],
        w: &MoeWeights,
        h: usize,
        e: usize,
        top_k: usize,
        inter: usize,
        shexp_inter: usize,
    ) -> Vec<f32> {
        let mut logits = vec![0.0f32; e];
        for (ei, logit) in logits.iter_mut().enumerate() {
            let mut s = 0.0;
            for (hi, &xv) in x.iter().enumerate() {
                s += xv * w.router[hi * e + ei];
            }
            *logit = s;
        }
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();

        let mut idx: Vec<usize> = (0..e).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let top = &idx[..top_k];
        let top_sum: f32 = top.iter().map(|&i| probs[i]).sum();

        let mut moe_out = vec![0.0f32; h];
        for &ei in top {
            let wgt = probs[ei] / top_sum;
            let mut act = vec![0.0f32; inter];
            for (ii, a) in act.iter_mut().enumerate() {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for (hi, &xv) in x.iter().enumerate() {
                    let row_base = (ei * h + hi) * 2 * inter;
                    g += xv * w.w_in[row_base + ii];
                    u += xv * w.w_in[row_base + inter + ii];
                }
                *a = silu(g) * u;
            }
            for (ho, mo) in moe_out.iter_mut().enumerate() {
                let mut s = 0.0f32;
                for (ii, &a) in act.iter().enumerate() {
                    s += a * w.w_out[(ei * inter + ii) * h + ho];
                }
                *mo += wgt * s;
            }
        }

        let mut sact = vec![0.0f32; shexp_inter];
        for (ii, a) in sact.iter_mut().enumerate() {
            let mut g = 0.0f32;
            let mut u = 0.0f32;
            for (hi, &xv) in x.iter().enumerate() {
                g += xv * w.sg[hi * shexp_inter + ii];
                u += xv * w.su[hi * shexp_inter + ii];
            }
            *a = silu(g) * u;
        }
        let mut shexp = vec![0.0f32; h];
        for (ho, sh) in shexp.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for (ii, &a) in sact.iter().enumerate() {
                s += a * w.sd[ii * h + ho];
            }
            *sh = s;
        }
        let gate_logit: f32 = x.iter().zip(w.sgi.iter()).map(|(&xv, &sv)| xv * sv).sum();
        let gate = sigmoid_ref(gate_logit);

        (0..h).map(|ho| moe_out[ho] + gate * shexp[ho]).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_ffn_ref_rows(
        x: &[Vec<f32>],
        w: &MoeWeights,
        h: usize,
        e: usize,
        top_k: usize,
        inter: usize,
        shexp_inter: usize,
    ) -> Vec<Vec<f32>> {
        x.iter()
            .map(|row| moe_ffn_ref(row, w, h, e, top_k, inter, shexp_inter))
            .collect()
    }

    /// SC-006: the graph tracer's `poot-eval` output matches the independent reference within `1e-4` at
    /// every position. `E=4`, `top_k=2` exercises both a selected and a dropped expert; `L=3` covers the
    /// `L>1` `moe_grouped` routing path (not only the `L=1` `moe_sparse` decode path).
    #[test]
    fn qwen38_moe_ffn_matches_independent_reference() {
        let (h, e, top_k, inter, shexp_inter, l) = (4, 4, 2, 3, 2, 3);
        let g = trace_moe_probe(h, e, top_k, inter, shexp_inter, l);
        let x: Vec<Vec<f32>> = (0..l).map(|t| fill(h, seed_of(&format!("x{t}")))).collect();
        let w = synth_weights(h, e, inter, shexp_inter);

        let got = eval_probe(&g, &x, &w, l, h);
        let want = moe_ffn_ref_rows(&x, &w, h, e, top_k, inter, shexp_inter);

        for t in 0..l {
            for c in 0..h {
                let (a, b) = (got[t][c], want[t][c]);
                assert!(
                    (a - b).abs() < 1e-4,
                    "position {t}, channel {c}: got {a}, want {b}"
                );
            }
        }
    }

    /// A deliberate mutation breaks the differential test and reverting it passes again. The mutation
    /// runs the independent reference with `top_k - 1` while the traced graph keeps `top_k`, so the
    /// reference selects one fewer expert and renormalizes over a different set.
    #[test]
    fn mutation_breaks_then_revert_restores_qwen38_moe_ffn_match() {
        let (h, e, top_k, inter, shexp_inter, l) = (4, 4, 2, 3, 2, 3);
        let g = trace_moe_probe(h, e, top_k, inter, shexp_inter, l);
        let x: Vec<Vec<f32>> = (0..l).map(|t| fill(h, seed_of(&format!("x{t}")))).collect();
        let w = synth_weights(h, e, inter, shexp_inter);
        let got = eval_probe(&g, &x, &w, l, h);

        let mutated_want = moe_ffn_ref_rows(&x, &w, h, e, top_k - 1, inter, shexp_inter);
        let mut mismatched = false;
        'outer: for t in 0..l {
            for c in 0..h {
                if (got[t][c] - mutated_want[t][c]).abs() >= 1e-4 {
                    mismatched = true;
                    break 'outer;
                }
            }
        }
        assert!(mismatched, "mutated top_k should break the match");

        let restored_want = moe_ffn_ref_rows(&x, &w, h, e, top_k, inter, shexp_inter);
        for t in 0..l {
            for c in 0..h {
                assert!(
                    (got[t][c] - restored_want[t][c]).abs() < 1e-4,
                    "reverted comparison should match again at position {t}, channel {c}"
                );
            }
        }
    }
}

/// Independent-reference differential test for the Hyper-Connections composition
/// ([`qwen38_gated_residual`]/[`qwen38_gated_residual_inject`]/[`qwen38_gated_residual_mixer`]), in the
/// pattern of `moe_ffn_oracle`: a small standalone graph (a widened `[.., hc_count*H]` input, wrap a
/// sublayer, collapse), checked against a hand-written pure-Rust reference of
/// `Qwen4ExpTextGatedResidual.forward`. It runs the round trip of [`trace_qwen38_prefill`]'s per-layer
/// loop (one `use_combine=True` call wrapping a sub-layer, then one `use_combine=False` collapse), with
/// a plain `Linear` standing in for the attention/GDN/MLP sub-layer.
mod gated_residual_oracle {
    use super::*;

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 0.3
            })
            .collect()
    }

    /// Real-magnitude synthetic weights, as `cpu_oracle::weight`: RMSNorm gammas near `1.0`, everything
    /// else near `0.0`.
    fn weight(name: &str, n: usize, is_norm_gamma: bool) -> Vec<f32> {
        let raw = fill(n, seed_of(name));
        if is_norm_gamma {
            raw.iter().map(|v| 1.0 + v * 0.05).collect()
        } else {
            raw.iter().map(|v| v * 0.1).collect()
        }
    }

    struct GrWeights {
        hc_norm: Vec<f32>,        // [hc_count*H]
        mix_down: Vec<f32>,       // [hc_count*H, lowrank]
        mix_up: Vec<f32>,         // [lowrank, hc_count*H]
        inject: Option<Vec<f32>>, // [hc_count*H, hc_count], None for use_combine=False
    }

    fn gr_weights(
        prefix: &str,
        hc_count: usize,
        h: usize,
        lowrank: usize,
        with_inject: bool,
    ) -> GrWeights {
        let hc_h = hc_count * h;
        GrWeights {
            hc_norm: weight(&format!("{prefix}.hc_norm"), hc_h, true),
            mix_down: weight(&format!("{prefix}.mix_down"), hc_h * lowrank, false),
            mix_up: weight(&format!("{prefix}.mix_up"), lowrank * hc_h, false),
            inject: with_inject
                .then(|| weight(&format!("{prefix}.inject"), hc_h * hc_count, false)),
        }
    }

    /// Standalone graph: `x = probe.hyper_input` -> [`qwen38_gated_residual`] (`probe.attn`) -> a plain
    /// `Linear` sub-layer (`probe.sublayer.weight`) -> [`qwen38_gated_residual_inject`] ->
    /// [`qwen38_gated_residual_mixer`] (`probe.mixer`) -> output `[1, L, H]`. No embedding, attention,
    /// GDN, MLP, or `lm_head`.
    fn trace_gr_probe(hc_count: usize, h: usize, lowrank: usize, l: usize, eps: f32) -> Graph {
        let b = Builder::new();
        let x = b.constant(
            "probe.hyper_input",
            TensorType::f32(vec![1, l, hc_count * h]),
        );
        let (mixed_input, inj_w) =
            qwen38_gated_residual(&b, x, "probe.attn", hc_count, h, lowrank, eps);
        let w_sub = b.constant("probe.sublayer.weight", TensorType::f32(vec![h, h]));
        let sublayer_out = linear(&b, mixed_input, w_sub, None);
        let x2 = qwen38_gated_residual_inject(&b, x, sublayer_out, inj_w);
        let out = qwen38_gated_residual_mixer(&b, x2, "probe.mixer", hc_count, h, lowrank, eps);
        b.finish(out)
    }

    fn eval_probe(
        g: &Graph,
        x: &[Vec<f32>],
        attn_w: &GrWeights,
        sub_w: &[f32],
        mixer_w: &GrWeights,
        l: usize,
        h: usize,
    ) -> Vec<Vec<f32>> {
        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let name = meta.name.as_deref().expect("const without a name");
            let data: Vec<f32> = if name == "probe.hyper_input" {
                x.iter().flatten().copied().collect()
            } else if name == "probe.sublayer.weight" {
                sub_w.to_vec()
            } else if name == "probe.attn.hc_norm.weight" {
                attn_w.hc_norm.clone()
            } else if name == "probe.attn.input_mix_weight_down.weight" {
                attn_w.mix_down.clone()
            } else if name == "probe.attn.input_mix_weight_up.weight" {
                attn_w.mix_up.clone()
            } else if name == "probe.attn.block_inject_weight.weight" {
                attn_w.inject.clone().expect("attn must have inject")
            } else if name == "probe.mixer.hc_norm.weight" {
                mixer_w.hc_norm.clone()
            } else if name == "probe.mixer.input_mix_weight_down.weight" {
                mixer_w.mix_down.clone()
            } else if name == "probe.mixer.input_mix_weight_up.weight" {
                mixer_w.mix_up.clone()
            } else {
                panic!("unexpected constant {name}")
            };
            inputs.insert(
                id,
                poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data).into(),
            );
        }
        let out = poot_eval::eval(
            g,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("cpu eval")
        .output
        .into_host()
        .expect("dense output");
        (0..l)
            .map(|t| out.as_f32().unwrap()[t * h..(t + 1) * h].to_vec())
            .collect()
    }

    fn silu(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    fn sigmoid_ref(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    /// `w` is `[in_dim, out_dim]` (the `poot_graph_ir::ops::linear` in-major convention).
    fn linear_ref(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += x[i] * w[i * out_dim + o];
            }
            y[o] = acc;
        }
        y
    }

    /// `Qwen4ExpTextRMSNorm(group_size=H)`'s reshape-normalize-flatten (see [`qwen38_grouped_rmsnorm`]):
    /// each `H`-sized group normalizes independently, then the full `[hc_count*H]` weight multiplies
    /// elementwise.
    fn grouped_rmsnorm_ref(x: &[f32], w: &[f32], hc_count: usize, h: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; hc_count * h];
        for g in 0..hc_count {
            let seg = &x[g * h..(g + 1) * h];
            let ms: f32 = seg.iter().map(|v| v * v).sum::<f32>() / h as f32;
            let den = (ms + eps).sqrt();
            for i in 0..h {
                out[g * h + i] = seg[i] / den;
            }
        }
        for (o, &wv) in out.iter_mut().zip(w.iter()) {
            *o *= wv;
        }
        out
    }

    /// `Qwen4ExpTextGatedResidual.forward`'s shared prefix (see [`qwen38_gated_residual_core`]): returns
    /// `(hyper_input_normed, mixed_input)`.
    fn gated_residual_core_ref(
        x: &[f32],
        w: &GrWeights,
        hc_count: usize,
        h: usize,
        lowrank: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let hc_h = hc_count * h;
        let normed = grouped_rmsnorm_ref(x, &w.hc_norm, hc_count, h, eps);
        let down = linear_ref(&normed, &w.mix_down, hc_h, lowrank);
        let down_act: Vec<f32> = down.iter().map(|&v| silu(v / hc_count as f32)).collect();
        let up = linear_ref(&down_act, &w.mix_up, lowrank, hc_h);
        let gate: Vec<f32> = up.iter().map(|&v| sigmoid_ref(v)).collect();

        let mut mixed = vec![0.0f32; h];
        for g in 0..hc_count {
            for i in 0..h {
                mixed[i] += gate[g * h + i] * normed[g * h + i];
            }
        }
        for v in mixed.iter_mut() {
            *v /= hc_count as f32;
        }
        (normed, mixed)
    }

    /// `use_combine=True`: returns `(mixed_input, injection_weights)`. `inject_scale` is `2.0` for the
    /// real composition; the mutation test overrides it.
    fn gated_residual_ref(
        x: &[f32],
        w: &GrWeights,
        hc_count: usize,
        h: usize,
        lowrank: usize,
        eps: f32,
        inject_scale: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let hc_h = hc_count * h;
        let (normed, mixed) = gated_residual_core_ref(x, w, hc_count, h, lowrank, eps);
        let inject_logit = linear_ref(
            &normed,
            w.inject.as_ref().expect("use_combine=True needs inject"),
            hc_h,
            hc_count,
        );
        let injw: Vec<f32> = inject_logit
            .iter()
            .map(|&v| inject_scale * sigmoid_ref(v / hc_count as f32))
            .collect();
        (mixed, injw)
    }

    /// `hidden_states = hyper_input + injection.flatten(-2)` (see [`qwen38_gated_residual_inject`]).
    fn gated_residual_inject_ref(
        hyper_input: &[f32],
        sublayer_out: &[f32],
        inject_w: &[f32],
        hc_count: usize,
        h: usize,
    ) -> Vec<f32> {
        let mut out = hyper_input.to_vec();
        for g in 0..hc_count {
            for i in 0..h {
                out[g * h + i] += sublayer_out[i] * inject_w[g];
            }
        }
        out
    }

    /// `use_combine=False`: returns just `mixed_input`.
    fn gated_residual_mixer_ref(
        x: &[f32],
        w: &GrWeights,
        hc_count: usize,
        h: usize,
        lowrank: usize,
        eps: f32,
    ) -> Vec<f32> {
        let (_normed, mixed) = gated_residual_core_ref(x, w, hc_count, h, lowrank, eps);
        mixed
    }

    /// The full round trip [`trace_gr_probe`]'s graph runs per token position: wrap a plain `Linear`
    /// sub-layer in Hyper-Connections, then collapse. `inject_scale` lets the mutation test diverge from
    /// the real `2.0`.
    #[allow(clippy::too_many_arguments)]
    fn full_ref(
        x_row: &[f32],
        attn_w: &GrWeights,
        sub_w: &[f32],
        mixer_w: &GrWeights,
        hc_count: usize,
        h: usize,
        lowrank: usize,
        eps: f32,
        inject_scale: f32,
    ) -> Vec<f32> {
        let (mixed, injw) =
            gated_residual_ref(x_row, attn_w, hc_count, h, lowrank, eps, inject_scale);
        let sub_out = linear_ref(&mixed, sub_w, h, h);
        let x2 = gated_residual_inject_ref(x_row, &sub_out, &injw, hc_count, h);
        gated_residual_mixer_ref(&x2, mixer_w, hc_count, h, lowrank, eps)
    }

    struct Fixture {
        g: Graph,
        x: Vec<Vec<f32>>,
        attn_w: GrWeights,
        sub_w: Vec<f32>,
        mixer_w: GrWeights,
        hc_count: usize,
        h: usize,
        lowrank: usize,
        eps: f32,
        l: usize,
    }

    fn build_fixture() -> Fixture {
        let (hc_count, h, lowrank, l, eps) = (3, 4, 5, 2, 1e-5);
        let g = trace_gr_probe(hc_count, h, lowrank, l, eps);
        let x: Vec<Vec<f32>> = (0..l)
            .map(|t| fill(hc_count * h, seed_of(&format!("x{t}"))))
            .collect();
        let attn_w = gr_weights("probe.attn", hc_count, h, lowrank, true);
        let sub_w = weight("probe.sublayer", h * h, false);
        let mixer_w = gr_weights("probe.mixer", hc_count, h, lowrank, false);
        Fixture {
            g,
            x,
            attn_w,
            sub_w,
            mixer_w,
            hc_count,
            h,
            lowrank,
            eps,
            l,
        }
    }

    /// The graph tracer's `poot-eval` output matches the independent reference within `1e-4` at every
    /// position. It would catch the residual add using the normed stream instead of the raw one (see
    /// [`qwen38_gated_residual`]): [`gated_residual_inject_ref`] adds back `x_row` (the original,
    /// pre-`hc_norm` argument), matching [`qwen38_gated_residual_inject`].
    #[test]
    fn qwen38_gated_residual_matches_independent_reference() {
        let f = build_fixture();
        let got = eval_probe(&f.g, &f.x, &f.attn_w, &f.sub_w, &f.mixer_w, f.l, f.h);
        let want: Vec<Vec<f32>> = f
            .x
            .iter()
            .map(|row| {
                full_ref(
                    row, &f.attn_w, &f.sub_w, &f.mixer_w, f.hc_count, f.h, f.lowrank, f.eps, 2.0,
                )
            })
            .collect();

        for t in 0..f.l {
            for c in 0..f.h {
                let (a, b) = (got[t][c], want[t][c]);
                assert!(
                    (a - b).abs() < 1e-4,
                    "position {t}, channel {c}: got {a}, want {b}"
                );
            }
        }
    }

    /// R474-004 / card 523b: the `hc_count`-axis reduce is a direct non-last-axis `Reduce`, not a
    /// transpose-to-trailing-axis dance (`compile`'s `lower_nonlast_reduces` legalizes a non-last-axis
    /// reduce for every target instead). The traced graph carries no tracer `Transpose`
    /// equation, and at least one `Reduce` targets an axis that is not its input's trailing axis.
    #[test]
    fn qwen38_gated_residual_reduces_the_natural_axis_with_no_tracer_transpose() {
        let g = trace_gr_probe(2, 8, 3, 1, 1e-6);
        assert!(
            !g.eqns
                .iter()
                .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::Transpose { .. })),
            "the tracer must not transpose around a reduce any more"
        );
        let has_nonlast_reduce = g.eqns.iter().any(|eqn| {
            let poot_graph_ir::OpKind::Reduce { axis, .. } = &eqn.op else {
                return false;
            };
            let poot_graph_ir::Operand::Value(input) = eqn.inputs[0] else {
                panic!("reduce input must be a value");
            };
            axis + 1 != g.aval(input).shape.len()
        });
        assert!(
            has_nonlast_reduce,
            "the hc_count axis reduce should not be the trailing axis; this probe should still \
             exercise it"
        );
    }

    /// A deliberate mutation breaks the differential test and reverting it passes again. The mutation
    /// runs the independent reference with `inject_scale = 1.0` (dropping the leading `2.0` of the real
    /// `2 * sigmoid(..)`) while the traced graph keeps `2.0` baked into [`qwen38_gated_residual`],
    /// diverging in every injected channel.
    #[test]
    fn mutation_breaks_then_revert_restores_qwen38_gated_residual_match() {
        let f = build_fixture();
        let got = eval_probe(&f.g, &f.x, &f.attn_w, &f.sub_w, &f.mixer_w, f.l, f.h);

        let mutated_want: Vec<Vec<f32>> = f
            .x
            .iter()
            .map(|row| {
                full_ref(
                    row, &f.attn_w, &f.sub_w, &f.mixer_w, f.hc_count, f.h, f.lowrank, f.eps, 1.0,
                )
            })
            .collect();
        let mut mismatched = false;
        'outer: for t in 0..f.l {
            for c in 0..f.h {
                if (got[t][c] - mutated_want[t][c]).abs() >= 1e-4 {
                    mismatched = true;
                    break 'outer;
                }
            }
        }
        assert!(mismatched, "mutated inject_scale should break the match");

        let restored_want: Vec<Vec<f32>> = f
            .x
            .iter()
            .map(|row| {
                full_ref(
                    row, &f.attn_w, &f.sub_w, &f.mixer_w, f.hc_count, f.h, f.lowrank, f.eps, 2.0,
                )
            })
            .collect();
        for t in 0..f.l {
            for c in 0..f.h {
                assert!(
                    (got[t][c] - restored_want[t][c]).abs() < 1e-4,
                    "reverted comparison should match again at position {t}, channel {c}"
                );
            }
        }
    }
}

/// The N-gram Embedding / PLE layer (`Qwen4ExpTextNGramEmbedding`/`Qwen4ExpTextPLELayer`). Same
/// discipline as [`super::tests::cpu_oracle`] and [`super::tests::gated_residual_oracle`]: a standalone
/// probe graph of only the composition under test, cross-checked against an independent hand-written
/// Rust reference (a per-position loop and an explicit shift-multiply-add conv), plus a
/// mutation-and-revert check.
mod ple_oracle {
    use super::*;

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Real-magnitude synthetic weights, as `cpu_oracle::weight`: RMSNorm gammas near `1.0`, everything
    /// else near `0.0`.
    fn weight(name: &str, n: usize, is_norm_gamma: bool) -> Vec<f32> {
        let raw = fill(n, seed_of(name));
        if is_norm_gamma {
            raw.iter().map(|v| 1.0 + v * 0.05).collect()
        } else {
            raw.iter().map(|v| v * 0.1).collect()
        }
    }

    /// A tiny stand-in for `Qwen4ExpPleConfig`, shrunk where the real one is huge but structurally
    /// identical: `ngram_size = 3` (two n-gram orders, two-token context), a prime-per-head vocab split,
    /// and a dilated conv whose dilation is `ngram_size`. `ngram_vocab_size_base = 17` makes the four
    /// head vocab sizes the four primes above 16 (17, 19, 23, 29), hand-checkable (see
    /// [`ngram_tables_match_hand_derivation`]) and far below `poot_eval`'s f32 `Gather`-index
    /// exactness limit, unlike the real 20000000 base.
    fn tiny_ple() -> Qwen4ExpPleConfig {
        Qwen4ExpPleConfig {
            // The real value, kept one-indexed, so this fixture pins the off-by-one that
            // `Qwen4ExpPleConfig::ple_layer_index` resolves (real `[2]` -> 0-based decoder layer 1).
            ple_layer_ids: vec![2],
            ple_embed_dim: 8,
            ple_conv_kernel_size: 3,
            ngram_size: 3,
            ngram_vocab_size_base: 17,
            heads_per_ngram: 2,
            make_ngram_vocab_size_divisible_by: 8,
            seed: 1234,
            eos_token_id: 5,
            vocab_size: 11,
        }
    }

    /// A token window with an EOS in the middle, so the reference and the port both must get the
    /// `_shift_right_ignore_eos` segment reset right (an n-gram must not straddle the EOS).
    fn tiny_tokens() -> Vec<i32> {
        vec![3, 7, 5, 2, 9, 1]
    }

    struct PleWeights {
        table: Vec<f32>,      // [padded_vocab, head_dim_per_ngram]
        key_proj: Vec<f32>,   // [ple_embed_dim, hc_count*H]
        value_proj: Vec<f32>, // [ple_embed_dim, H]
        norm_key: Vec<f32>,   // [hc_count*H]
        norm_query: Vec<f32>, // [hc_count*H]
        norm_conv: Vec<f32>,  // [hc_count*H]
        conv1d: Vec<f32>,     // [ple_conv_kernel_size, hc_count*H]
    }

    fn ple_weights(
        ple: &Qwen4ExpPleConfig,
        tables: &Qwen4ExpNgramTables,
        hc_count: usize,
        h: usize,
    ) -> PleWeights {
        let hc_h = hc_count * h;
        let e = ple.ple_embed_dim;
        PleWeights {
            table: weight(
                "probe.ple.table",
                tables.padded_vocab_size * ple.head_dim_per_ngram(),
                false,
            ),
            key_proj: weight("probe.ple.key_proj", e * hc_h, false),
            value_proj: weight("probe.ple.value_proj", e * h, false),
            norm_key: weight("probe.ple.norm_key", hc_h, true),
            norm_query: weight("probe.ple.norm_query", hc_h, true),
            norm_conv: weight("probe.ple.norm_conv", hc_h, true),
            conv1d: weight("probe.ple.conv1d", ple.ple_conv_kernel_size * hc_h, false),
        }
    }

    fn bind_weights(w: &PleWeights) -> HashMap<String, Vec<f32>> {
        let mut m = HashMap::new();
        m.insert(
            "probe.ple.ple_embedding.ngram_embedding.weight".to_string(),
            w.table.clone(),
        );
        m.insert("probe.ple.key_proj.weight".to_string(), w.key_proj.clone());
        m.insert(
            "probe.ple.value_proj.weight".to_string(),
            w.value_proj.clone(),
        );
        m.insert("probe.ple.norm_key.weight".to_string(), w.norm_key.clone());
        m.insert(
            "probe.ple.norm_query.weight".to_string(),
            w.norm_query.clone(),
        );
        m.insert(
            "probe.ple.norm_conv.weight".to_string(),
            w.norm_conv.clone(),
        );
        m.insert("probe.ple.conv1d.weight".to_string(), w.conv1d.clone());
        m
    }

    /// Standalone prefill probe: `probe.hidden` (a `[1, L, hc_count*H]` widened stream) plus a
    /// `probe.ple.ngram_ids` `I32` id block -> [`qwen38_ple_prefill`] -> the decoder layer's
    /// `hidden_states + ple(..)` add. No embedding, attention, Hyper-Connections, or `lm_head`.
    fn trace_ple_probe(
        ple: &Qwen4ExpPleConfig,
        tables: &Qwen4ExpNgramTables,
        hc_count: usize,
        h: usize,
        l: usize,
        eps: f32,
    ) -> Graph {
        let b = Builder::new();
        let x = b.constant("probe.hidden", TensorType::f32(vec![1, l, hc_count * h]));
        let ids = b.constant(
            "probe.ple.ngram_ids",
            TensorType::new(vec![l, ple.ngram_heads()], DType::I32),
        );
        let out = qwen38_ple_prefill(&b, x, ids, "probe", ple, tables, hc_count, h, eps);
        let summed = b.binary(BinOp::Add, x, out);
        b.finish(summed)
    }

    /// [`trace_ple_probe`]'s decode counterpart: one token, one carried `probe.ple.conv_cache` state
    /// pair, so the decode-vs-prefill test can drive an incremental trajectory through
    /// [`qwen38_ple_decode`].
    fn trace_ple_probe_decode(
        ple: &Qwen4ExpPleConfig,
        tables: &Qwen4ExpNgramTables,
        hc_count: usize,
        h: usize,
        eps: f32,
    ) -> Graph {
        let b = Builder::new();
        let x = b.constant("probe.hidden", TensorType::f32(vec![1, 1, hc_count * h]));
        let ids = b.constant(
            "probe.ple.ngram_ids",
            TensorType::new(vec![1, ple.ngram_heads()], DType::I32),
        );
        let cc_in = b.state_input(
            "probe.ple.conv_cache",
            TensorType::f32(vec![1, ple.conv_state_len(), hc_count * h]),
            StateRole::Recurrent,
        );
        let (out, cc_out) =
            qwen38_ple_decode(&b, x, ids, cc_in, "probe", ple, tables, hc_count, h, eps);
        let summed = b.binary(BinOp::Add, x, out);
        b.finish_with_state(summed, &[(cc_in, cc_out)])
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_ple_probe(
        g: &Graph,
        x: &[Vec<f32>],
        ids: &[i32],
        w: &HashMap<String, Vec<f32>>,
        state: &[poot_tensor::HostTensor],
        l: usize,
        hc_h: usize,
    ) -> (Vec<Vec<f32>>, Vec<poot_tensor::HostTensor>) {
        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            match &meta.storage {
                Storage::State => continue, // bound below, in g.state order
                Storage::Const => {}
                other => panic!("unexpected storage {other:?} in a PLE probe"),
            }
            let name = meta.name.as_deref().expect("const without a name");
            let t = if name == "probe.ple.ngram_ids" {
                poot_tensor::HostTensor::i32(meta.aval.shape.clone(), ids.to_vec())
            } else {
                let data = if name == "probe.hidden" {
                    x.iter().flatten().copied().collect()
                } else {
                    w.get(name)
                        .unwrap_or_else(|| panic!("no weight bound for {name}"))
                        .clone()
                };
                poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data)
            };
            inputs.insert(id, t.into());
        }
        for (ci, &(si, _)) in g.state.iter().enumerate() {
            let t = state
                .get(ci)
                .cloned()
                .unwrap_or_else(|| poot_tensor::HostTensor::zeros(g.aval(si).shape.clone()));
            inputs.insert(si, t.into());
        }
        let step = poot_eval::eval(
            g,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("cpu eval");
        let out = step.output.into_host().expect("dense output");
        let new_state: Vec<poot_tensor::HostTensor> = step
            .state
            .into_iter()
            .map(|v| v.into_host().expect("dense state"))
            .collect();
        let rows = (0..l)
            .map(|t| out.as_f32().unwrap()[t * hc_h..(t + 1) * hc_h].to_vec())
            .collect();
        (rows, new_state)
    }

    fn silu(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    fn sigmoid_ref(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    /// `torch.sign`: `-1`, `0`, or `+1` (not `f32::signum`, which returns `+1.0` at `+0.0`).
    fn torch_sign(v: f32) -> f32 {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    }

    /// `w` is `[in_dim, out_dim]` (the `poot_graph_ir::ops::linear` in-major convention).
    fn linear_ref(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; out_dim];
        for (o, slot) in y.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += x[i] * w[i * out_dim + o];
            }
            *slot = acc;
        }
        y
    }

    /// `Qwen4ExpTextRMSNorm(hc_count*H, group_size=H)`, re-derived here rather than imported from
    /// `gated_residual_oracle`'s private helper, so this test does not depend on another test module.
    fn grouped_rmsnorm_ref(x: &[f32], w: &[f32], hc_count: usize, h: usize, eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; hc_count * h];
        for g in 0..hc_count {
            let seg = &x[g * h..(g + 1) * h];
            let ms: f32 = seg.iter().map(|v| v * v).sum::<f32>() / h as f32;
            let den = (ms + eps).sqrt();
            for i in 0..h {
                out[g * h + i] = seg[i] / den * w[g * h + i];
            }
        }
        out
    }

    /// Independent hand-written reference for `Qwen4ExpTextPLELayer.forward`
    /// (`modeling_qwen4_exp.py` lines 1241-1261), returning the PLE contribution per position (the
    /// caller adds it to the stream). A per-position loop plus an explicit shift-multiply-add dilated
    /// conv, not the graph's reshape/broadcast/`Ge`-sign formulation. `gate_floor` is the real
    /// `clamp_min(1e-6)`; the mutation test overrides it.
    #[allow(clippy::too_many_arguments)]
    fn ple_ref(
        x: &[Vec<f32>],
        ids: &[i32],
        w: &PleWeights,
        ple: &Qwen4ExpPleConfig,
        hc_count: usize,
        h: usize,
        eps: f32,
        gate_floor: f32,
    ) -> Vec<Vec<f32>> {
        let l = x.len();
        let hc_h = hc_count * h;
        let e = ple.ple_embed_dim;
        let heads = ple.ngram_heads();
        let dh = ple.head_dim_per_ngram();

        let mut gated: Vec<Vec<f32>> = Vec::with_capacity(l);
        let mut gated_normed: Vec<Vec<f32>> = Vec::with_capacity(l);
        for (t, x_row) in x.iter().enumerate() {
            // embedding lookup + flatten(-2)
            let mut emb = vec![0.0f32; e];
            for g in 0..heads {
                let row = ids[t * heads + g] as usize;
                emb[g * dh..(g + 1) * dh].copy_from_slice(&w.table[row * dh..(row + 1) * dh]);
            }
            let key = linear_ref(&emb, &w.key_proj, e, hc_h);
            let key_normed = grouped_rmsnorm_ref(&key, &w.norm_key, hc_count, h, eps);
            let value = linear_ref(&emb, &w.value_proj, e, h);
            let query_normed = grouped_rmsnorm_ref(x_row, &w.norm_query, hc_count, h, eps);

            let mut row = vec![0.0f32; hc_h];
            for g in 0..hc_count {
                let mut dot = 0.0f32;
                for i in 0..h {
                    dot += key_normed[g * h + i] * query_normed[g * h + i];
                }
                let gate = dot / (h as f32).sqrt();
                let signed = gate.abs().max(gate_floor).sqrt() * torch_sign(gate);
                let s = sigmoid_ref(signed);
                for i in 0..h {
                    row[g * h + i] = s * value[i];
                }
            }
            gated_normed.push(grouped_rmsnorm_ref(&row, &w.norm_conv, hc_count, h, eps));
            gated.push(row);
        }

        // Dilated causal depthwise conv over the normed copy, then SiLU, then add the un-normed one.
        let k = ple.ple_conv_kernel_size;
        let d = ple.conv_dilation();
        (0..l)
            .map(|t| {
                (0..hc_h)
                    .map(|c| {
                        let mut acc = 0.0f32;
                        for j in 0..k {
                            let back = (k - 1 - j) * d;
                            if t >= back {
                                acc += w.conv1d[j * hc_h + c] * gated_normed[t - back][c];
                            }
                        }
                        gated[t][c] + silu(acc)
                    })
                    .collect()
            })
            .collect()
    }

    struct Fixture {
        ple: Qwen4ExpPleConfig,
        tables: Qwen4ExpNgramTables,
        w: PleWeights,
        bound: HashMap<String, Vec<f32>>,
        ids: Vec<i32>,
        x: Vec<Vec<f32>>,
        tokens: Vec<i32>,
        hc_count: usize,
        h: usize,
        l: usize,
        eps: f32,
    }

    fn build_fixture() -> Fixture {
        let (hc_count, h, eps) = (2, 4, 1e-5);
        let ple = tiny_ple();
        let tables = qwen38_ngram_tables(&ple, 0);
        let tokens = tiny_tokens();
        let l = tokens.len();
        let ctx = qwen38_ngram_previous_context(&ple, &[]);
        let ids = qwen38_ngram_ids(&ple, &tables, &ctx, &tokens).expect("context is context_len");
        let w = ple_weights(&ple, &tables, hc_count, h);
        let bound = bind_weights(&w);
        let raw = fill(l * hc_count * h, seed_of("probe.hidden"));
        let x = (0..l)
            .map(|t| {
                raw[t * hc_count * h..(t + 1) * hc_count * h]
                    .iter()
                    .map(|v| v * 0.5)
                    .collect()
            })
            .collect();
        Fixture {
            ple,
            tables,
            w,
            bound,
            ids,
            x,
            tokens,
            hc_count,
            h,
            l,
            eps,
        }
    }

    /// The derived tables match a hand-derivation: with `ngram_vocab_size_base = 17`, the four hash heads
    /// take the four primes strictly above `base - 1 = 16` (`_find_nth_prime_after(base-1, head+1)`),
    /// offsets are their running sum, and the row count rounds up to the
    /// `make_ngram_vocab_size_divisible_by` multiple.
    #[test]
    fn ngram_tables_match_hand_derivation() {
        let ple = tiny_ple();
        let t = qwen38_ngram_tables(&ple, 0);
        assert_eq!(t.head_vocab_sizes, vec![17, 19, 23, 29]);
        assert_eq!(t.head_offsets, vec![0, 17, 36, 59]);
        assert_eq!(t.total_vocab_size, 88);
        assert_eq!(t.padded_vocab_size, 88); // 88 is already a multiple of 8
        assert_eq!(t.layer_multipliers.len(), ple.ngram_size);
        // `_build_layer_multipliers` emits odd multipliers (`2 * (splitmix % bound) + 1`).
        for m in &t.layer_multipliers {
            assert_eq!(m % 2, 1, "layer multipliers must be odd");
            assert!(*m > 0);
        }
    }

    /// Golden vectors produced by a verbatim Python transcription of the real
    /// `_splitmix64`/`_build_layer_multipliers`/`_find_nth_prime_after` and
    /// `Qwen4ExpTextNGramEmbedding.forward` hash (module-level functions copied unchanged, torch tensor
    /// ops rewritten as plain lists with explicit int64 wraparound) and pasted here. This is the one
    /// check in this module independent of the Rust code: if [`qwen38_ngram_ids`]'s wrapping `int64`
    /// multiply, the `torch.remainder` sign convention, or the EOS segment rule drifted, these numbers
    /// would not reproduce.
    #[test]
    fn ngram_hash_matches_verbatim_python_golden_vectors() {
        let ple = tiny_ple();
        let tables = qwen38_ngram_tables(&ple, 0);
        assert_eq!(
            tables.layer_multipliers,
            vec![73077407744147129, 749479866669413493, 395950012578265759]
        );

        let ctx = qwen38_ngram_previous_context(&ple, &[]);
        assert_eq!(ctx, vec![ple.eos_token_id; ple.context_len()]);
        let ids =
            qwen38_ngram_ids(&ple, &tables, &ctx, &tiny_tokens()).expect("context is context_len");
        assert_eq!(
            ids,
            vec![
                8, 24, 36, 77, 15, 22, 55, 59, 0, 31, 45, 74, 7, 32, 42, 81, 12, 19, 53, 81, 7, 19,
                51, 86
            ]
        );

        // The same golden run at the real config's values (`vocab_size = 248320`, `seed = 1234`,
        // `ngram_size = 3`, `ngram_vocab_size_base = 20000000`, `heads_per_ngram = 8`,
        // `make_ngram_vocab_size_divisible_by = 128`), pinning the derivation at real scale. The padded
        // row count reproduces the checkpoint's 128 x `[2500012, 160]` shards: 128 * 2500012 = 320001536.
        let real = Qwen4ExpPleConfig {
            ple_layer_ids: vec![2],
            ple_embed_dim: 2560,
            ple_conv_kernel_size: 4,
            ngram_size: 3,
            ngram_vocab_size_base: 20_000_000,
            heads_per_ngram: 8,
            make_ngram_vocab_size_divisible_by: 128,
            seed: 1234,
            eos_token_id: 248044,
            vocab_size: 248320,
        };
        let rt = qwen38_ngram_tables(&real, 0);
        assert_eq!(
            rt.layer_multipliers,
            vec![23703573157769, 20109073645365, 8052911324071]
        );
        assert_eq!(rt.total_vocab_size, 320001446);
        assert_eq!(rt.padded_vocab_size, 320001536);
        assert_eq!(rt.padded_vocab_size, 128 * 2500012);
        assert_eq!(real.ngram_heads(), 16);
        assert_eq!(real.head_dim_per_ngram(), 160);
        assert_eq!(real.conv_state_len(), 9);
        assert_eq!(real.ple_layer_index(1), Some(0));
    }

    /// `ple_layer_ids` is one-indexed (`configuration_qwen4_exp.py:240-246`), so the real `[2]` selects
    /// 0-based decoder layer 1; pinned here.
    #[test]
    fn ple_layer_ids_are_one_indexed() {
        let ple = tiny_ple();
        assert_eq!(ple.ple_layer_index(0), None);
        assert_eq!(ple.ple_layer_index(1), Some(0));
        assert_eq!(ple.ple_layer_index(2), None);
    }

    /// An independently-formulated n-gram id reference: finds each position's segment start by scanning
    /// backward for the nearest preceding EOS (not the port's forward scan or the real `cummax`), and
    /// mixes in `i128` before truncating to 64 bits (not `i64::wrapping_mul`), with a
    /// `((m % s) + s) % s` non-negative modulo instead of `rem_euclid`. Same answers, three mechanisms.
    fn ngram_ids_ref(
        ple: &Qwen4ExpPleConfig,
        tables: &Qwen4ExpNgramTables,
        previous_context: &[i32],
        input_ids: &[i32],
    ) -> Vec<i32> {
        let eos = ple.eos_token_id as i64;
        let heads = ple.ngram_heads();
        let per = ple.heads_per_ngram;
        let history: Vec<i64> = previous_context
            .iter()
            .chain(input_ids.iter())
            .map(|&v| v as i64)
            .collect();
        let n = history.len();

        let segment_start = |t: usize| -> usize {
            // the position just after the nearest EOS strictly before t (real `previous_eos` is the
            // one-position-shifted cummax, so token t's own EOS does not start a new segment at t).
            (0..t)
                .rev()
                .find(|&u| history[u] == eos)
                .map_or(0, |u| u + 1)
        };
        let shifted: Vec<Vec<i64>> = (0..ple.ngram_size)
            .map(|shift| {
                (0..n)
                    .map(|t| {
                        if shift == 0 {
                            return history[t];
                        }
                        match t.checked_sub(shift) {
                            Some(src) if src >= segment_start(t) => history[src],
                            _ => eos,
                        }
                    })
                    .collect()
            })
            .collect();

        let wrap = |v: i128| -> i64 { (v & ((1i128 << 64) - 1)) as u64 as i64 };
        let mut ids = vec![0i32; n * heads];
        for ngram in 2..=ple.ngram_size {
            let start_idx = (ngram - 2) * per;
            for t in 0..n {
                let mut mixed = wrap(shifted[0][t] as i128 * tables.layer_multipliers[0] as i128);
                for (position, row) in shifted.iter().enumerate().take(ngram).skip(1) {
                    mixed ^= wrap(row[t] as i128 * tables.layer_multipliers[position] as i128);
                }
                for j in 0..per {
                    let head = start_idx + j;
                    let s = tables.head_vocab_sizes[head];
                    ids[t * heads + head] =
                        (((mixed % s) + s) % s + tables.head_offsets[head]) as i32;
                }
            }
        }
        ids[(n - input_ids.len()) * heads..].to_vec()
    }

    /// [`qwen38_ngram_ids`] matches [`ngram_ids_ref`] on a window with an EOS in the middle (which forces
    /// the segment-reset rule to matter) and on an EOS-free window.
    #[test]
    fn ngram_ids_match_independent_reference() {
        let ple = tiny_ple();
        let tables = qwen38_ngram_tables(&ple, 0);
        for tokens in [tiny_tokens(), vec![3, 7, 8, 2, 9, 1], vec![5, 5, 5, 1]] {
            let ctx = qwen38_ngram_previous_context(&ple, &[]);
            let got =
                qwen38_ngram_ids(&ple, &tables, &ctx, &tokens).expect("context is context_len");
            let want = ngram_ids_ref(&ple, &tables, &ctx, &tokens);
            assert_eq!(got, want, "tokens {tokens:?}");
            assert_eq!(got.len(), tokens.len() * ple.ngram_heads());
            // Every id must land inside its own head's slice of the shared table.
            for (t, _) in tokens.iter().enumerate() {
                for head in 0..ple.ngram_heads() {
                    let id = got[t * ple.ngram_heads() + head] as i64;
                    let lo = tables.head_offsets[head];
                    let hi = lo + tables.head_vocab_sizes[head];
                    assert!((lo..hi).contains(&id), "id {id} outside head {head}");
                }
            }
        }
    }

    /// The EOS segment rule is observable: a token before an EOS must not leak into the n-gram of a
    /// token after it. Position 3 (just past the EOS at position 2) gets the same ids as if the
    /// sequence had started there, while position 4 (two past it) does not (its 2-gram history is the
    /// real token at position 3).
    #[test]
    fn ngram_ids_do_not_cross_an_eos_boundary() {
        let ple = tiny_ple();
        let tables = qwen38_ngram_tables(&ple, 0);
        let heads = ple.ngram_heads();
        let ctx = qwen38_ngram_previous_context(&ple, &[]);

        let with_prefix = qwen38_ngram_ids(&ple, &tables, &ctx, &[3, 7, 5, 2, 9])
            .expect("context is context_len");
        let other_prefix = qwen38_ngram_ids(&ple, &tables, &ctx, &[8, 1, 5, 2, 9])
            .expect("context is context_len");
        // Positions 3 and 4 sit in the segment that starts right after the EOS at index 2, so their ids
        // cannot depend on anything before it.
        for t in [3usize, 4] {
            assert_eq!(
                with_prefix[t * heads..(t + 1) * heads],
                other_prefix[t * heads..(t + 1) * heads],
                "position {t} must not see across the EOS"
            );
        }
        // Position 1 IS inside the first segment, so a different position-0 token must change it.
        assert_ne!(
            with_prefix[heads..2 * heads],
            other_prefix[heads..2 * heads]
        );
    }

    /// The graph tracer's `poot-eval` output matches the independent reference within `1e-4` at every
    /// position (embedding lookup, three grouped RMSNorms, the signed-sqrt gate, and the dilated
    /// depthwise causal conv).
    #[test]
    fn qwen38_ple_prefill_matches_independent_reference() {
        let f = build_fixture();
        let g = trace_ple_probe(&f.ple, &f.tables, f.hc_count, f.h, f.l, f.eps);
        g.validate().expect("PLE probe should validate");
        let hc_h = f.hc_count * f.h;
        let (got, _) = eval_ple_probe(&g, &f.x, &f.ids, &f.bound, &[], f.l, hc_h);
        let contrib = ple_ref(&f.x, &f.ids, &f.w, &f.ple, f.hc_count, f.h, f.eps, 1e-6);

        for t in 0..f.l {
            for c in 0..hc_h {
                let want = f.x[t][c] + contrib[t][c];
                assert!(
                    (got[t][c] - want).abs() < 1e-4,
                    "position {t}, channel {c}: got {}, want {want}",
                    got[t][c]
                );
            }
        }
    }

    /// A deliberate mutation breaks the differential test and reverting it passes again. The mutation
    /// runs the independent reference with the `gate.abs().clamp_min(1e-6)` floor raised to `0.25`,
    /// changing every gate below that magnitude, while the traced graph keeps `1e-6` in
    /// `qwen38_ple_core_with_embeddings`.
    #[test]
    fn mutation_breaks_then_revert_restores_qwen38_ple_match() {
        let f = build_fixture();
        let g = trace_ple_probe(&f.ple, &f.tables, f.hc_count, f.h, f.l, f.eps);
        let hc_h = f.hc_count * f.h;
        let (got, _) = eval_ple_probe(&g, &f.x, &f.ids, &f.bound, &[], f.l, hc_h);

        let mutated = ple_ref(&f.x, &f.ids, &f.w, &f.ple, f.hc_count, f.h, f.eps, 0.25);
        let mut mismatched = false;
        'outer: for t in 0..f.l {
            for c in 0..hc_h {
                if (got[t][c] - (f.x[t][c] + mutated[t][c])).abs() >= 1e-4 {
                    mismatched = true;
                    break 'outer;
                }
            }
        }
        assert!(mismatched, "a mutated gate floor should break the match");

        let restored = ple_ref(&f.x, &f.ids, &f.w, &f.ple, f.hc_count, f.h, f.eps, 1e-6);
        for t in 0..f.l {
            for c in 0..hc_h {
                let want = f.x[t][c] + restored[t][c];
                assert!(
                    (got[t][c] - want).abs() < 1e-4,
                    "reverted comparison should match again at position {t}, channel {c}"
                );
            }
        }
    }

    /// [`qwen38_ple_decode`] driven step by step from an empty conv cache reproduces
    /// [`qwen38_ple_prefill`]'s whole-sequence output. This pins the dilated conv's decode-side cache
    /// depth (`(K-1)*dilation`, not `K-1`) and shift order. Each step re-derives its `[1, ngram_heads]`
    /// id row from the token history, as a `Runner` integration would (see
    /// [`qwen38_ngram_previous_context`]).
    #[test]
    fn qwen38_ple_decode_loop_matches_prefill() {
        let f = build_fixture();
        let hc_h = f.hc_count * f.h;
        let pg = trace_ple_probe(&f.ple, &f.tables, f.hc_count, f.h, f.l, f.eps);
        let (want, _) = eval_ple_probe(&pg, &f.x, &f.ids, &f.bound, &[], f.l, hc_h);

        let dg = trace_ple_probe_decode(&f.ple, &f.tables, f.hc_count, f.h, f.eps);
        dg.validate().expect("PLE decode probe should validate");
        let mut state: Vec<poot_tensor::HostTensor> = Vec::new();
        for (t, want_row) in want.iter().enumerate() {
            let ctx = qwen38_ngram_previous_context(&f.ple, &f.tokens[..t]);
            let ids = qwen38_ngram_ids(&f.ple, &f.tables, &ctx, &f.tokens[t..t + 1])
                .expect("context is context_len");
            // The per-step id row must agree with the whole-sequence prefill block.
            assert_eq!(
                ids,
                f.ids[t * f.ple.ngram_heads()..(t + 1) * f.ple.ngram_heads()],
                "step {t} ids"
            );
            let row = vec![f.x[t].clone()];
            let (got, new_state) = eval_ple_probe(&dg, &row, &ids, &f.bound, &state, 1, hc_h);
            state = new_state;
            for c in 0..hc_h {
                assert!(
                    (got[0][c] - want_row[c]).abs() < 1e-4,
                    "step {t}, channel {c}: decode {} vs prefill {}",
                    got[0][c],
                    want_row[c]
                );
            }
        }
    }

    /// The whole-model tracers accept a `Some(..)` PLE config and place exactly one PLE block on the
    /// 0-based decoder layer the one-indexed `ple_layer_ids` names, checked structurally by the
    /// `{layer}.ple.*` constants and the extra decode state pair the PLE conv cache adds.
    fn ple_model_cfg(ple: Option<Qwen4ExpPleConfig>) -> Qwen4ExpModelConfig {
        Qwen4ExpModelConfig {
            cfg: tiny_cfg(),
            qcfg: tiny_qcfg(4),
            gdn: Qwen4ExpGdnConfig {
                num_k_heads: 1,
                num_v_heads: 2,
                head_dim: 2,
                conv_k: 4,
            },
            layer_is_full: vec![false, false, false, true],
            vocab: 6,
            max_pos: 16,
            ffn_inter: 8,
            moe_n_experts: 3,
            moe_top_k: 2,
            moe_inter: 4,
            eps: 1e-5,
            chunk: 4,
            hc_count: 2,
            hc_lowrank: 3,
            ple,
        }
    }

    fn const_names(g: &Graph) -> Vec<String> {
        g.inputs
            .iter()
            .filter_map(|&id| g.meta(id).name.clone())
            .collect()
    }

    #[test]
    fn trace_qwen38_prefill_places_ple_on_the_one_indexed_layer() {
        // `tiny_cfg().hidden = 8`, `hc_count = 2`, so the PLE embed dim must stay divisible by its 4 hash
        // heads; 8 works and keeps `head_dim_per_ngram = 2`.
        let ple = tiny_ple();
        let with_ple = trace_qwen38_prefill(&ple_model_cfg(Some(ple.clone())), 8);
        with_ple
            .validate()
            .expect("PLE-enabled prefill trace should validate");
        let names = const_names(&with_ple);
        assert!(
            names
                .iter()
                .any(|n| n == "activation.layers.1.ple.ngram_ids")
        );
        assert!(names.iter().any(|n| n == "layers.1.ple.conv1d.weight"));
        for li in [0usize, 2, 3] {
            assert!(
                !names
                    .iter()
                    .any(|n| n.starts_with(&format!("layers.{li}.ple."))),
                "layer {li} must not carry a PLE block"
            );
        }

        // Disabling it must leave the graph unchanged.
        let without = trace_qwen38_prefill(&ple_model_cfg(None), 8);
        assert!(!const_names(&without).iter().any(|n| n.contains(".ple.")));
        assert_eq!(with_ple.state.len(), without.state.len());
    }

    #[test]
    fn trace_qwen38_decode_places_ple_and_its_conv_cache() {
        let ple = tiny_ple();
        let with_ple = trace_qwen38_decode(&ple_model_cfg(Some(ple)), 8);
        with_ple
            .validate()
            .expect("PLE-enabled decode trace should validate");
        let names = const_names(&with_ple);
        assert!(
            names
                .iter()
                .any(|n| n == "activation.layers.1.ple.ngram_ids")
        );
        assert!(names.iter().any(|n| n == "layers.1.ple.conv_cache"));

        let without = trace_qwen38_decode(&ple_model_cfg(None), 8);
        // The PLE adds exactly one extra carried state pair (its dilated short-conv cache).
        assert_eq!(with_ple.state.len(), without.state.len() + 1);
    }

    /// Three non-degenerate token histories for [`Qwen4ExpNgramHistory`]: one with an EOS mid-sequence
    /// (the segment-reset rule must matter across chunk boundaries), one shorter than
    /// `context_len = 2` (the EOS-pad fill path; first token 4 so its first id row differs from the
    /// EOS-mid case under the same EOS prior), and one long enough that every chunk split actually
    /// exercises the carried window. `tiny_ple`'s `eos_token_id` is 5, which the long history avoids so
    /// carry (not reset) is what makes its mid-sequence ids correct.
    fn history_cases() -> Vec<(&'static str, Vec<i32>)> {
        vec![
            ("eos_mid", tiny_tokens()),
            ("shorter_than_context", vec![4]),
            (
                "long_no_eos",
                vec![1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            ),
        ]
    }

    /// Several ways to split `len` tokens into consecutive chunk sizes that sum to `len`, always
    /// including the whole-sequence prefill and size-1 chunks.
    fn chunk_splits(len: usize) -> Vec<Vec<usize>> {
        assert!(len > 0, "histories under test are non-empty");
        let mut splits = vec![vec![len]];
        if len >= 3 {
            let mut threes = vec![3usize; len / 3];
            let rem = len % 3;
            if rem > 0 {
                threes.push(rem);
            }
            splits.push(threes);
        }
        splits.push(vec![1; len]);
        if len >= 2 {
            splits.push(vec![1, len - 1]);
            splits.push(vec![len - 1, 1]);
        }
        if len >= 4 {
            splits.push(vec![1, 2, len - 3]);
        }
        splits
    }

    /// Reference: one full prefill through [`qwen38_ngram_ids`] over the whole sequence with the
    /// from-scratch context. The carrier must reproduce this for every chunk split.
    fn full_prefill_ids(
        ple: &Qwen4ExpPleConfig,
        tables: &Qwen4ExpNgramTables,
        tokens: &[i32],
    ) -> Vec<i32> {
        let ctx = qwen38_ngram_previous_context(ple, &[]);
        qwen38_ngram_ids(ple, tables, &ctx, tokens).expect("from-scratch context is context_len")
    }

    /// Feed `tokens` through one [`Qwen4ExpNgramHistory`] as consecutive chunks of the given sizes
    /// (prefill and/or decode steps), concatenating the id rows.
    fn ids_via_history(
        ple: &Qwen4ExpPleConfig,
        tables: &Qwen4ExpNgramTables,
        tokens: &[i32],
        chunk_sizes: &[usize],
        reset_between_chunks: bool,
    ) -> Vec<i32> {
        let mut history = Qwen4ExpNgramHistory::new(ple);
        let mut ids = Vec::new();
        let mut pos = 0usize;
        for &size in chunk_sizes {
            let end = pos + size;
            assert!(end <= tokens.len(), "chunk split overruns the sequence");
            if reset_between_chunks {
                // Mutation path: drop the carried window between chunks (fresh EOS context each time).
                history = Qwen4ExpNgramHistory::new(ple);
            }
            let chunk = &tokens[pos..end];
            ids.extend(
                history
                    .ids_for_chunk(ple, tables, chunk)
                    .expect("carrier window is context_len"),
            );
            pos = end;
        }
        assert_eq!(pos, tokens.len(), "chunk split must cover the sequence");
        ids
    }

    /// Card 449 SC-003 foundation / Card 363 two-token carry: for each of the three histories, ids
    /// from one full prefill equal ids from chunked prefill (several splits, including size-1) and from
    /// token-by-token decode, including mixed prefix-prefill + suffix-decode. The full-prefill core on
    /// the whole sequence is the reference.
    #[test]
    fn qwen38_ngram_history_chunked_prefill_and_decode_match_full_prefill() {
        let ple = tiny_ple();
        let tables = qwen38_ngram_tables(&ple, 0);
        let heads = ple.ngram_heads();

        for (name, tokens) in history_cases() {
            let reference = full_prefill_ids(&ple, &tables, &tokens);
            assert_eq!(
                reference.len(),
                tokens.len() * heads,
                "{name}: full prefill must emit one id row per token"
            );

            for sizes in chunk_splits(tokens.len()) {
                let got = ids_via_history(&ple, &tables, &tokens, &sizes, false);
                assert_eq!(got, reference, "{name}: chunk sizes {sizes:?}");
            }

            // Token-by-token decode is the size-1 split, called out separately as the decode path.
            let decode = ids_via_history(&ple, &tables, &tokens, &vec![1; tokens.len()], false);
            assert_eq!(decode, reference, "{name}: token-by-token decode");

            // Mixed runner shape: prefill a non-empty prefix in one chunk, decode the rest stepwise.
            if tokens.len() >= 2 {
                for split_at in 1..tokens.len() {
                    let mut sizes = vec![split_at];
                    sizes.extend(std::iter::repeat_n(1, tokens.len() - split_at));
                    let mixed = ids_via_history(&ple, &tables, &tokens, &sizes, false);
                    assert_eq!(
                        mixed, reference,
                        "{name}: prefill first {split_at} then decode"
                    );
                }
            }
        }
    }

    /// The three histories are non-degenerate: full-prefill id vectors differ pairwise (SC-003's
    /// "ids vary across at least three non-degenerate token histories"), multi-token rows are not all
    /// identical within a history (a hard-coded single id row fails), and the two EOS-context first
    /// rows differ when their first tokens differ (a hard-coded first row fails).
    #[test]
    fn qwen38_ngram_history_three_histories_produce_distinct_ids() {
        let ple = tiny_ple();
        let tables = qwen38_ngram_tables(&ple, 0);
        let heads = ple.ngram_heads();
        let cases: Vec<_> = history_cases()
            .into_iter()
            .map(|(name, tokens)| {
                let ids = full_prefill_ids(&ple, &tables, &tokens);
                (name, tokens, ids)
            })
            .collect();
        assert_eq!(cases.len(), 3);
        for (name, _, ids) in &cases {
            assert!(!ids.is_empty(), "{name}: empty id block is degenerate");
            if *name != "shorter_than_context" {
                let row0 = &ids[..heads];
                let all_same = ids.chunks(heads).all(|row| row == row0);
                assert!(
                    !all_same,
                    "{name}: every id row identical — ids did not vary across positions"
                );
            }
        }
        // Pairwise: different lengths already distinguish the short case; for equal-length pairs the
        // value comparison does the work. Cross-check first rows for the two multi-token histories
        // (both start with EOS prior context, so only the first token can make row 0 differ).
        let eos = &cases[0].2;
        let long = &cases[2].2;
        assert_ne!(
            &eos[..heads],
            &long[..heads],
            "same EOS prior context but different first tokens must hash differently"
        );
        let short = &cases[1].2;
        assert_ne!(
            &eos[..heads],
            &short[..heads],
            "first tokens 3 vs 4 under EOS prior must hash differently"
        );
        assert_ne!(
            cases[0].2, cases[1].2,
            "eos_mid vs shorter_than_context id blocks must differ"
        );
        assert_ne!(
            cases[0].2, cases[2].2,
            "eos_mid vs long_no_eos id blocks must differ"
        );
        assert_ne!(
            cases[1].2, cases[2].2,
            "shorter_than_context vs long_no_eos id blocks must differ"
        );
    }

    /// Mutation row (recorded): carrying a previous_context window one token short of `context_len`
    /// is rejected with the typed error — this is the path that used to `assert_eq!`-panic.
    #[test]
    fn mutation_history_window_one_token_too_short_is_rejected() {
        let ple = tiny_ple();
        let tables = qwen38_ngram_tables(&ple, 0);
        let ctx = qwen38_ngram_previous_context(&ple, &[]);
        assert_eq!(ctx.len(), ple.context_len());
        let short = &ctx[..ctx.len() - 1];
        let tokens = tiny_tokens();
        let err = qwen38_ngram_ids(&ple, &tables, short, &tokens)
            .expect_err("a window one token too short must not be accepted");
        assert_eq!(
            err,
            Qwen4ExpNgramError::ContextLength {
                got: ple.context_len() - 1,
                want: ple.context_len(),
            }
        );
        // The carrier itself never presents a short window: it starts at context_len and only
        // advances through `qwen38_ngram_previous_context`.
        let mut history = Qwen4ExpNgramHistory::new(&ple);
        assert_eq!(history.previous_context().len(), ple.context_len());
        history
            .ids_for_chunk(&ple, &tables, &tokens[..1])
            .expect("carrier keeps a full window");
        assert_eq!(history.previous_context().len(), ple.context_len());
    }

    /// Mutation row (recorded): resetting the history between chunks (fresh EOS context each time
    /// instead of carrying the trailing window) breaks full-prefill equivalence on a mid-sequence
    /// history that has not hit an EOS. The failure is on the first non-trivial chunk boundary; rows
    /// at or before a real EOS may still match by the segment-reset rule, which is why the fixture
    /// avoids EOS in the long case.
    #[test]
    fn mutation_reset_history_between_chunks_diverges_from_full_prefill() {
        let ple = tiny_ple();
        let tables = qwen38_ngram_tables(&ple, 0);
        let heads = ple.ngram_heads();
        let tokens = vec![1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
        let reference = full_prefill_ids(&ple, &tables, &tokens);

        // Split into halves: chunk 0 is correct under reset (from-scratch start), chunk 1 must see
        // tokens 7..9 as prior context and diverges when reset to EOS instead.
        let reset = ids_via_history(&ple, &tables, &tokens, &[8, 8], true);
        assert_ne!(
            reset, reference,
            "resetting history between chunks must not reproduce the full-prefill ids"
        );
        // Localize the failure: the first chunk still matches (it starts from the same from-scratch
        // context either way); the second chunk's first row is where carry is required.
        let per_row = heads;
        assert_eq!(
            &reset[..8 * per_row],
            &reference[..8 * per_row],
            "first chunk under reset still matches a from-scratch prefill"
        );
        assert_ne!(
            &reset[8 * per_row..9 * per_row],
            &reference[8 * per_row..9 * per_row],
            "first row after the reset boundary diverges: got {:?}, want {:?}",
            &reset[8 * per_row..9 * per_row],
            &reference[8 * per_row..9 * per_row],
        );

        // Carrying (no reset) on the same split is the control: it must still match.
        let carried = ids_via_history(&ple, &tables, &tokens, &[8, 8], false);
        assert_eq!(carried, reference, "carried history on the same split");
    }

    /// The carrier starts from a non-empty prior with the same rule as
    /// [`qwen38_ngram_previous_context`], so resuming mid-sequence matches a full prefill of the
    /// remaining suffix under that prior.
    #[test]
    fn qwen38_ngram_history_from_prior_matches_core_on_suffix() {
        let ple = tiny_ple();
        let tables = qwen38_ngram_tables(&ple, 0);
        let prior = vec![9, 1, 2];
        let suffix = vec![4, 6, 7, 8];
        let ctx = qwen38_ngram_previous_context(&ple, &prior);
        let reference =
            qwen38_ngram_ids(&ple, &tables, &ctx, &suffix).expect("context is context_len");

        let mut history = Qwen4ExpNgramHistory::from_prior(&ple, &prior);
        assert_eq!(history.previous_context(), ctx.as_slice());
        let mut got = Vec::new();
        for size in [2, 2] {
            let start = got.len() / ple.ngram_heads();
            let end = start + size;
            got.extend(
                history
                    .ids_for_chunk(&ple, &tables, &suffix[start..end])
                    .expect("carrier window is context_len"),
            );
        }
        assert_eq!(got, reference);
    }
}

fn real_ple() -> Qwen4ExpPleConfig {
    Qwen4ExpPleConfig {
        ple_layer_ids: vec![2],
        ple_embed_dim: 2560,
        ple_conv_kernel_size: 4,
        ngram_size: 3,
        ngram_vocab_size_base: 20_000_000,
        heads_per_ngram: 8,
        make_ngram_vocab_size_divisible_by: 128,
        seed: 1234,
        eos_token_id: 248044,
        vocab_size: 248320,
    }
}

fn encode_hist(ple: &Qwen4ExpPleConfig, raw: &[i32]) -> Vec<i32> {
    raw.iter().map(|&t| t ^ ple.eos_token_id).collect()
}

fn eval_ngram_graph(
    g: &Graph,
    tokens: &[i32],
    history_raw: &[i32],
    ple: &Qwen4ExpPleConfig,
) -> (Vec<i32>, Vec<i32>) {
    let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        match &meta.storage {
            Storage::Slot(Slot::Token) => {
                assert_eq!(meta.aval.shape, vec![tokens.len()]);
                inputs.insert(
                    id,
                    poot_tensor::HostTensor::i32(meta.aval.shape.clone(), tokens.to_vec()).into(),
                );
            }
            Storage::State => {
                assert_eq!(meta.aval.shape, vec![1, ple.context_len()]);
                inputs.insert(
                    id,
                    poot_tensor::HostTensor::i32(
                        meta.aval.shape.clone(),
                        encode_hist(ple, history_raw),
                    )
                    .into(),
                );
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }
    let step = poot_eval::eval(
        g,
        &inputs,
        poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
    )
    .expect("cpu eval ngram graph");
    let ids = step.output.into_host().expect("dense output");
    let state: Vec<poot_tensor::HostTensor> = step
        .state
        .into_iter()
        .map(|v| v.into_host().expect("dense state"))
        .collect();
    let id_words = ids.as_i32().expect("ids are I32").to_vec();
    let state_words = state[0].as_i32().expect("state is I32").to_vec();
    (id_words, state_words)
}

fn decode_hist(ple: &Qwen4ExpPleConfig, encoded: &[i32]) -> Vec<i32> {
    encoded.iter().map(|&t| t ^ ple.eos_token_id).collect()
}

/// In-graph ids match the host [`qwen38_ngram_ids`] bit-exactly at the REAL constants for three
/// non-degenerate histories (fresh EOS pad, nonzero prior, and EOS inside the carried window).
#[test]
fn graph_ngram_ids_match_host_at_real_constants_three_histories() {
    let ple = real_ple();
    let tables = qwen38_ngram_tables(&ple, 0);
    let tokens: Vec<i32> = vec![11, 248044, 7, 99, 12345, 6];
    // Three non-degenerate histories: fresh, nonzero, EOS-inside-window.
    let histories: [Vec<i32>; 3] = [
        vec![ple.eos_token_id, ple.eos_token_id],
        vec![3, 8],
        vec![42, ple.eos_token_id],
    ];
    let mut produced = Vec::new();
    for (hi, hist) in histories.iter().enumerate() {
        let g = trace_qwen38_ngram_ids(&ple, &tables, tokens.len());
        let (got, next) = eval_ngram_graph(&g, &tokens, hist, &ple);
        let want = qwen38_ngram_ids(&ple, &tables, hist, &tokens).expect("host ids");
        assert_eq!(got, want, "history {hi} {hist:?}");
        // New history = last context_len of (hist ++ tokens), raw.
        let mut combined = hist.clone();
        combined.extend_from_slice(&tokens);
        let want_next: Vec<i32> = combined[combined.len() - ple.context_len()..].to_vec();
        let got_next = decode_hist(&ple, &next);
        assert_eq!(got_next, want_next, "history {hi} state");
        produced.push(got.clone());
    }
    // The three histories must actually produce distinct ids (non-degenerate).
    assert_ne!(
        produced[0], produced[1],
        "fresh vs nonzero history ids differ"
    );
    assert_ne!(
        produced[1], produced[2],
        "nonzero vs EOS-window history ids differ"
    );
    // Hard-coding one id block would fail: every row of each block varies within the block too.
    assert!(
        produced[0]
            .chunks(ple.ngram_heads())
            .take(3)
            .collect::<Vec<_>>()
            .windows(2)
            .any(|w| w[0] != w[1]),
        "ids vary across positions"
    );
}

/// Chunked prefill (two pieces) plus decode equal one full prefill through the carried state,
/// matching the host at every step.
#[test]
fn graph_ngram_ids_chunked_prefill_and_decode_equal_full_prefill() {
    let ple = real_ple();
    let tables = qwen38_ngram_tables(&ple, 0);
    // EOS exactly on the first chunk boundary (index 3 == end of the length-3 first chunk).
    let full: Vec<i32> = vec![10, 20, 30, ple.eos_token_id, 50, 60, 70];
    let ctx0 = vec![ple.eos_token_id; ple.context_len()];

    // Full prefill in one step.
    let g_full = trace_qwen38_ngram_ids(&ple, &tables, full.len());
    let (full_ids, _full_next) = eval_ngram_graph(&g_full, &full, &ctx0, &ple);
    let full_want = qwen38_ngram_ids(&ple, &tables, &ctx0, &full).expect("host full");
    assert_eq!(full_ids, full_want, "full prefill vs host");

    // Chunk 1: first 3, fresh state. Chunk 2: next 4, carried state. Decode: last 1 one at a
    // time after that.
    let g1 = trace_qwen38_ngram_ids(&ple, &tables, 3);
    let (ids1, st1_raw) = eval_ngram_graph(&g1, &full[..3], &ctx0, &ple);
    let ctx1 = decode_hist(&ple, &st1_raw);

    let g2 = trace_qwen38_ngram_ids(&ple, &tables, 4);
    let (ids2, st2_raw) = eval_ngram_graph(&g2, &full[3..7], &ctx1, &ple);
    let ctx2 = decode_hist(&ple, &st2_raw);

    let mut chunked = ids1.clone();
    chunked.extend_from_slice(&ids2);
    assert_eq!(chunked, full_ids, "chunked prefill equals full prefill");

    // Host cross-check of the chunks against the same carried contexts.
    let host1 = qwen38_ngram_ids(&ple, &tables, &ctx0, &full[..3]).unwrap();
    let host2 = qwen38_ngram_ids(&ple, &tables, &ctx1, &full[3..7]).unwrap();
    assert_eq!(ids1, host1, "chunk 1 vs host");
    assert_eq!(ids2, host2, "chunk 2 vs host");

    // Decode the remaining tokens one by one from ctx2 (already at end of full); append one more
    // synthetic step to prove decode from carried state.
    let decode_tok = [999];
    let g_d = trace_qwen38_ngram_ids(&ple, &tables, 1);
    let (ids_d, st_d) = eval_ngram_graph(&g_d, &decode_tok, &ctx2, &ple);
    let host_d = qwen38_ngram_ids(&ple, &tables, &ctx2, &decode_tok).unwrap();
    assert_eq!(ids_d, host_d, "decode step vs host");
    // The decode step must advance the carried window to [ctx2[1], 999].
    let want_st_d = vec![ctx2[1], decode_tok[0]];
    assert_eq!(
        decode_hist(&ple, &st_d),
        want_st_d,
        "decode-step state must be the advanced two-token window"
    );

    // Zeroed state decodes as EOS history: graph with zero buffer equals fresh prefill.
    let g0 = trace_qwen38_ngram_ids(&ple, &tables, full.len());
    let mut zero_inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
    for &id in &g0.inputs {
        let meta = g0.meta(id);
        let shape = meta.aval.shape.clone();
        let words = match &meta.storage {
            Storage::Slot(Slot::Token) => full.clone(),
            Storage::State => vec![0i32; shape.iter().product()],
            other => panic!("unexpected storage {other:?}"),
        };
        zero_inputs.insert(id, poot_tensor::HostTensor::i32(shape, words).into());
    }
    let zero_ids = poot_eval::eval(
        &g0,
        &zero_inputs,
        poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
    )
    .expect("zero-state eval")
    .output
    .into_host()
    .expect("dense output");
    assert_eq!(
        zero_ids.as_i32().unwrap(),
        full_want.as_slice(),
        "zeroed state must decode as EOS-filled history"
    );
}

/// Falsifiable coverage for `qwen4exp_ple_sharded_row_unscaled`/`qwen4exp_ple_scaled_row`: the
/// bounded, synthetic half of the "Sharded embedding graph" section (the metadata partition arithmetic
/// is checked against the real pinned `model.safetensors.index.json` values in
/// `qwen4exp_ple_inventory_matches_exact_partition`).
///
/// Four shards, not one (SC-002): a single shard would let every cross-shard claim (ids on both sides
/// of a boundary, independent owner pointer identity, row coverage without gaps/overlap) pass by
/// construction. Hand-picked E4M3 bytes (see `dense_row_gather_e4m3.rs`'s module doc for why):
/// `0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E, 0x50` decode to the dyadic progression `1.0, 2.0, .., 8.0`,
/// hand-verifiable past a power of two, and printed (not just asserted distinct) before either
/// falsifiable claim below trusts them.
mod card363_ple_sharded_lookup {
    use std::sync::Arc;

    use poot_eval::fp8::e4m3fn_tensor;
    use poot_eval::{EvalBudget, EvalOptions, Value as EvalValue, eval};
    use poot_graph_ir::ValueId;
    use poot_quant::scalar::e4m3fn_to_f32;
    use poot_tensor::HostTensor as EvalTensor;

    use super::fold_dense_bf16_row_gathers;
    use poot_load::packed_safetensors::{ExactSourceOwner, TensorDisposition};

    use super::super::*;
    use crate::test_support::safetensors::{SourceRow, load_checkpoint};

    const ROW_WIDTH: usize = 1;
    /// Row `2*shard + j` decodes to `(2*shard + j + 1) as f32` (see the module doc's byte list). Shard
    /// `s` owns rows `[2*s, 2*s + 2)`.
    const SHARD_BYTES: [[u8; 2]; 4] = [[0x38, 0x40], [0x44, 0x48], [0x4A, 0x4C], [0x4E, 0x50]];

    fn shard_ranges() -> Vec<Qwen4ExpPleShardRange> {
        (0..SHARD_BYTES.len())
            .map(|shard| Qwen4ExpPleShardRange {
                linear_id: format!("card363_test.ple_shard_{shard}"),
                rows: (shard * 2)..(shard * 2 + 2),
            })
            .collect()
    }

    /// The Card 359 front door: `ExactSourceOwner`'s fields are private even inside `poot-load`, so this
    /// is the only way outside that crate to get a real one (as Card 405's `dense_row_gather_e4m3.rs`).
    fn shard_owners() -> Vec<Arc<ExactSourceOwner>> {
        let rows = shard_ranges()
            .iter()
            .zip(SHARD_BYTES)
            .map(|(range, bytes)| {
                (
                    SourceRow {
                        name: range.linear_id.clone(),
                        dtype: "F8_E4M3",
                        shape: vec![range.rows.len(), ROW_WIDTH],
                        bytes: bytes.to_vec(),
                    },
                    TensorDisposition::StandaloneE4m3,
                )
            })
            .collect();
        let loaded = load_checkpoint("local/card363-ple-sharded-lookup-fixture", "0", b"{}", rows);
        assert_eq!(
            loaded.mixed.exact_metadata.len(),
            SHARD_BYTES.len(),
            "one owner per shard"
        );
        // Looked up by linear id, not by map iteration order (card 540a:
        // `exact_metadata` is keyed by tensor name): callers `zip` this with `shard_ranges()` and
        // need shard `i`'s owner at position `i`.
        shard_ranges()
            .iter()
            .map(|range| {
                Arc::clone(
                    &loaded
                        .mixed
                        .exact_metadata
                        .get(range.linear_id.as_str())
                        .unwrap_or_else(|| panic!("no owner for shard {}", range.linear_id))
                        .owner,
                )
            })
            .collect()
    }

    fn decoded_values() -> Vec<f32> {
        SHARD_BYTES
            .iter()
            .flat_map(|row| row.iter().map(|&byte| e4m3fn_to_f32(byte)))
            .collect()
    }

    fn value_id_by_name(graph: &Graph, name: &str) -> ValueId {
        graph
            .inputs
            .iter()
            .copied()
            .find(|&id| graph.meta(id).name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no graph input named {name}"))
    }

    /// Build the (unscaled, or scaled by `scale`) sharded-lookup graph, fold it through
    /// `fold_dense_bf16_row_gathers`, and return it plus the id input's
    /// `ValueId` and a closure that binds one id value and every shard owner (rebuilt per call; it
    /// only clones `Arc`s) for `eval`.
    fn build_folded(scale: Option<f32>) -> (Graph, ValueId, impl Fn(i32) -> EvalValue) {
        let owners = shard_owners();
        let b = Builder::new();
        let id = b.slot(Slot::Token, TensorType::scalar(DType::I32));
        let ranges = shard_ranges();
        let row = match scale {
            None => qwen4exp_ple_sharded_row_unscaled(&b, id, &ranges, ROW_WIDTH),
            Some(_) => {
                let scale_t = b.constant("card363_test.ple_scale", TensorType::scalar(DType::F32));
                qwen4exp_ple_scaled_row(&b, id, &ranges, ROW_WIDTH, scale_t)
            }
        };
        let graph = b.finish(row);
        let folded = fold_dense_bf16_row_gathers(&graph);
        let dense_row_gathers = folded
            .eqns
            .iter()
            .filter(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::DenseRowGather { .. }))
            .count();
        assert_eq!(
            dense_row_gathers,
            ranges.len(),
            "one DenseRowGather per shard branch must survive folding - if this is 0, the fold pass \
             did not recognize the Gather+Cast chain this composition builds, and the eval below would \
             exercise the ordinary generic E4M3Fn Gather dispatch instead of Card 405's own primitive"
        );
        let id_value = id.id;
        let owner_bindings: Vec<(ValueId, Arc<ExactSourceOwner>)> = ranges
            .iter()
            .zip(owners)
            .map(|(range, owner)| (value_id_by_name(&folded, &range.linear_id), owner))
            .collect();
        let folded_for_closure = folded.clone();
        let binder = move |id_value_i32: i32| -> EvalValue {
            let mut inputs = std::collections::HashMap::new();
            for (value_id, owner) in &owner_bindings {
                inputs.insert(
                    *value_id,
                    EvalValue::Host(
                        e4m3fn_tensor(owner.descriptor().shape().to_vec(), owner.bytes().to_vec())
                            .expect("fixture owner bytes match the declared shape"),
                    ),
                );
            }
            if let Some(scale) = scale {
                let scale_id = value_id_by_name(&folded_for_closure, "card363_test.ple_scale");
                inputs.insert(
                    scale_id,
                    EvalValue::Host(EvalTensor::f32(vec![], vec![scale])),
                );
            }
            inputs.insert(
                id_value,
                EvalValue::Host(EvalTensor::i32(vec![], vec![id_value_i32])),
            );
            eval(
                &folded_for_closure,
                &inputs,
                EvalOptions::new(EvalBudget::UNBOUNDED),
            )
            .expect("bounded sharded-lookup graph evaluates")
            .output
        };
        (folded, id_value, binder)
    }

    /// FR-009/SC-005 (mutation table row 1, gather half). Required red mutation: in
    /// `qwen4exp_ple_sharded_row_unscaled`, swap the shard-ordinal assignment of two shards (e.g. bind
    /// shard 1's table where shard 2's belongs); ids in the swapped shards' ranges then read the wrong
    /// shard's decoded value while other ids stay correct, changing exactly the swapped rows' expected
    /// values in this test and none of the others. Not run in this draft phase.
    #[test]
    fn qwen4exp_sharded_lookup_reads_owner_backed_rows() {
        let expected = decoded_values();
        eprintln!("qwen4exp_sharded_lookup_reads_owner_backed_rows: expected={expected:?}");
        assert_eq!(expected.len(), 8);
        for (i, &v) in expected.iter().enumerate() {
            assert!(v > 0.0, "row {i} decoded to non-positive {v}");
        }
        for pair in expected.windows(2) {
            assert_ne!(
                pair[0], pair[1],
                "adjacent rows must decode to distinct values"
            );
        }

        let (_graph, _id_value, eval_at) = build_folded(None);
        for id in 0..8i32 {
            let EvalValue::Host(tensor) = eval_at(id) else {
                panic!("unscaled sharded lookup must publish a dense F32 row");
            };
            eprintln!(
                "  id={id} -> output={:?} expected={}",
                tensor.as_f32().unwrap(),
                expected[id as usize]
            );
            assert_eq!(tensor.as_f32().unwrap(), [expected[id as usize]]);
        }
    }

    /// FR-009/SC-005 (mutation table row 1, scale half, and Card 405's distinguishability pattern one
    /// level up). Required red mutations, run separately and recorded, not built into an always-green
    /// assertion (mutation (a) edits this test's SCALE literal without touching the independent
    /// `expected` computation, which only a manual one-time desync can express):
    /// (a) scale-only: change only the `scale` literal `build_folded` embeds into the graph (e.g.
    ///     `Some(0.5)` -> `Some(0.75)`) while `expected`'s multiplier stays `0.5`; only this test's
    ///     assertions may change, and `qwen4exp_sharded_lookup_reads_owner_backed_rows` must stay
    ///     green (it has no scale).
    /// (b) gather-only: the row-swap mutation named on the sibling gather-alone test above; both this
    ///     test's and that test's assertions must change, since both read through
    ///     `qwen4exp_ple_sharded_row_unscaled`.
    #[test]
    fn qwen4exp_sharded_lookup_scales_only_gathered_rows() {
        const SCALE: f32 = 0.5;
        let expected: Vec<f32> = decoded_values().iter().map(|&v| v * SCALE).collect();
        eprintln!("qwen4exp_sharded_lookup_scales_only_gathered_rows: expected={expected:?}");

        let (_graph, _id_value, eval_at) = build_folded(Some(SCALE));
        for id in 0..8i32 {
            let EvalValue::Host(tensor) = eval_at(id) else {
                panic!("scaled sharded lookup must publish a dense F32 row");
            };
            eprintln!(
                "  id={id} -> output={:?} expected={}",
                tensor.as_f32().unwrap(),
                expected[id as usize]
            );
            assert_eq!(tensor.as_f32().unwrap(), [expected[id as usize]]);
        }
    }

    /// SC-002. Four shards, not one (see this module's doc: N=1 would fall into the
    /// collection-semantics trap for every claim below).
    #[test]
    fn qwen4exp_ple_shards_cover_rows_without_copy() {
        let ranges = shard_ranges();
        // Exact, contiguous, non-overlapping coverage of [0, 8), the bounded analog of the 128-way
        // tiling `qwen4exp_ple_shard_row_range` derives.
        let mut covered = 0usize;
        for range in &ranges {
            assert_eq!(
                range.rows.start, covered,
                "shard {} must start exactly where the previous shard ended",
                range.linear_id
            );
            covered = range.rows.end;
        }
        assert_eq!(covered, 8, "the four shards must cover [0, 8) exactly");

        let owners = shard_owners();
        for i in 0..owners.len() {
            for j in (i + 1)..owners.len() {
                assert!(
                    !Arc::ptr_eq(&owners[i], &owners[j]),
                    "shards {i} and {j} must not share one owner"
                );
            }
        }

        // Ids on both sides of every shard boundary (1|2, 3|4, 5|6) read their own, different shard.
        let (_graph, _id_value, eval_at) = build_folded(None);
        for &(left, right) in &[(1i32, 2i32), (3, 4), (5, 6)] {
            let EvalValue::Host(l) = eval_at(left) else {
                panic!("expected dense output")
            };
            let EvalValue::Host(r) = eval_at(right) else {
                panic!("expected dense output")
            };
            assert_ne!(
                l.as_f32().unwrap(),
                r.as_f32().unwrap(),
                "ids {left} and {right} cross a shard boundary and must read different shards"
            );
        }
    }

    /// SC-003 / Card 363 residual: a real-scale n-gram id ABOVE `2^24`, derived by the production
    /// host hash over the real config constants, bound end-to-end through the sharded PLE gather
    /// and evaluated on the CPU oracle against an independent reference. The id provably cannot
    /// survive an f32 round trip, so only an exact-I32 index path reads its row: the fixture puts
    /// the id and its f32-rounded neighbour in different shards with different bytes, and the
    /// expected value is the literal the id's own byte decodes to.
    #[test]
    fn qwen4exp_ngram_id_above_2_24_gathers_its_exact_row() {
        // Real-scale tables (`ngram_vocab_size_base = 20_000_000`): head 1 starts at the first
        // prime's offset, ~20M > 2^24, so every id in heads 1..15 is above 2^24 by construction -
        // no token-window luck involved.
        let ple = super::real_ple();
        let tables = qwen38_ngram_tables(&ple, 0);
        let ctx = qwen38_ngram_previous_context(&ple, &[]);
        let tokens = [11i32, 7, 99, 12_345];
        let ids = qwen38_ngram_ids(&ple, &tables, &ctx, &tokens).expect("host n-gram ids");
        let id = ids
            .iter()
            .copied()
            .find(|&value| value > (1 << 24))
            .expect("real-scale tables must produce an id above 2^24");
        let lossy = id as f32 as i32;
        assert_ne!(
            lossy, id,
            "this row's premise: id {id} does not survive an f32 round trip"
        );
        assert!(id > 0, "n-gram ids are nonnegative row indices");
        eprintln!("qwen4exp_ngram_id_above_2_24: id={id} lossy={lossy} ids={ids:?}");

        // The fixture bytes are the module's dyadic progression; print and pin them before any
        // claim below trusts the literals.
        assert_eq!(e4m3fn_to_f32(0x40), 2.0, "the id's own row");
        assert_eq!(e4m3fn_to_f32(0x38), 1.0, "every other row");

        // Two contiguous shards, split so the id and its rounded neighbour are never in the same
        // one, and so the id owns local row 0 of its shard.
        let id_us = id as usize;
        let lossy_us = lossy as usize;
        let (shard_rows, id_in_first): ([std::ops::Range<usize>; 2], bool) = if lossy < id {
            ([lossy_us..id_us, id_us..id_us + 1], false)
        } else {
            ([id_us..lossy_us, lossy_us..lossy_us + 1], true)
        };
        let ranges: Vec<Qwen4ExpPleShardRange> = shard_rows
            .iter()
            .enumerate()
            .map(|(ordinal, rows)| Qwen4ExpPleShardRange {
                linear_id: format!("card363_test.ple_above_2_24_shard_{ordinal}"),
                rows: rows.clone(),
            })
            .collect();
        let shard_owns_id = [id_in_first, !id_in_first];
        let rows: Vec<(SourceRow, TensorDisposition)> = ranges
            .iter()
            .enumerate()
            .map(|(ordinal, range)| {
                let bytes = (0..range.rows.len())
                    .map(|row| {
                        if shard_owns_id[ordinal] && row == 0 {
                            0x40
                        } else {
                            0x38
                        }
                    })
                    .collect();
                (
                    SourceRow {
                        name: range.linear_id.clone(),
                        dtype: "F8_E4M3",
                        shape: vec![range.rows.len(), ROW_WIDTH],
                        bytes,
                    },
                    TensorDisposition::StandaloneE4m3,
                )
            })
            .collect();
        let loaded = load_checkpoint("local/card363-ple-above-2-24-fixture", "0", b"{}", rows);
        assert_eq!(
            loaded.mixed.exact_metadata.len(),
            ranges.len(),
            "one owner per shard"
        );
        // `MixedLoadResult::exact_metadata` is keyed by tensor name (card 540a), so each
        // shard's owner is looked up by the linear id this test assigned it, not by a returned
        // row's position (which `load_checkpoint` never promised to preserve).
        let owners: Vec<Arc<ExactSourceOwner>> = ranges
            .iter()
            .map(|range| {
                Arc::clone(
                    &loaded
                        .mixed
                        .exact_metadata
                        .get(range.linear_id.as_str())
                        .unwrap_or_else(|| panic!("no owner for shard {}", range.linear_id))
                        .owner,
                )
            })
            .collect();
        for (ordinal, owner) in owners.iter().enumerate() {
            assert_eq!(
                owner.bytes()[0],
                if shard_owns_id[ordinal] { 0x40 } else { 0x38 },
                "owner for shard {ordinal} must be the shard this test assigned it"
            );
        }

        let b = Builder::new();
        let id_input = b.slot(Slot::Token, TensorType::scalar(DType::I32));
        let row = qwen4exp_ple_sharded_row_unscaled(&b, id_input, &ranges, ROW_WIDTH);
        let graph = b.finish(row);
        let folded = fold_dense_bf16_row_gathers(&graph);
        assert_eq!(
            folded
                .eqns
                .iter()
                .filter(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::DenseRowGather { .. }))
                .count(),
            ranges.len(),
            "one DenseRowGather per shard branch must survive folding"
        );

        let mut inputs = std::collections::HashMap::new();
        for (range, owner) in ranges.iter().zip(&owners) {
            let value_id = folded
                .inputs
                .iter()
                .copied()
                .find(|&value| folded.meta(value).name.as_deref() == Some(range.linear_id.as_str()))
                .unwrap_or_else(|| panic!("no graph input named {}", range.linear_id));
            inputs.insert(
                value_id,
                EvalValue::Host(
                    e4m3fn_tensor(owner.descriptor().shape().to_vec(), owner.bytes().to_vec())
                        .expect("fixture owner bytes match the declared shape"),
                ),
            );
        }
        // An I32-declared id bound as F32 is refused at bind, so no f32 round trip can happen for
        // this input even by accident - the refusal this row exists to pin.
        inputs.insert(
            id_input.id,
            EvalValue::Host(EvalTensor::f32(vec![], vec![id as f32])),
        );
        let refusal = eval(&folded, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect_err("an f32-mirrored index must be refused by the exact lane");
        assert!(
            format!("{refusal:?}").contains("Value::Host of a different dtype"),
            "unexpected refusal for an f32-mirrored index: {refusal:?}"
        );

        inputs.insert(
            id_input.id,
            EvalValue::Host(EvalTensor::i32(vec![], vec![id])),
        );
        let EvalValue::Host(output) =
            eval(&folded, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
                .expect("sharded-lookup graph over an id above 2^24 evaluates")
                .output
        else {
            panic!("sharded lookup must publish a dense F32 row");
        };
        eprintln!("  id={id} -> output={:?}", output.as_f32().unwrap());
        assert_eq!(
            output.as_f32().unwrap(),
            [2.0],
            "the exact path must read the id's own row (2.0); an f32-routed index would read \
             {lossy}'s row (1.0)"
        );
    }
}

/// SC-001/FR-002/FR-003: the 138-tensor partition arithmetic checked against
/// `model.safetensors.index.json` itself (SHA-256 `0419e2c2...`, 17,410,140 bytes, pinned in the card's
/// spec/evidence table), not a shrunk fixture. The literals below are the checkpoint's declared
/// metadata; the test proves the derived arithmetic (`QWEN4EXP_PLE_TOTAL_ROWS`/
/// `QWEN4EXP_PLE_E4M3_BYTES`/`QWEN4EXP_PLE_TOTAL_TENSORS`, each computed from smaller pinned constants)
/// reproduces them, so drift between the pinned per-shard numbers and the pinned totals is caught by
/// arithmetic.
mod card363_ple_inventory {
    use super::super::*;

    #[test]
    fn qwen4exp_ple_inventory_matches_exact_partition() {
        assert_eq!(QWEN4EXP_PLE_SHARDS, 128);
        assert_eq!(QWEN4EXP_PLE_ROWS_PER_SHARD, 2_500_012);
        assert_eq!(QWEN4EXP_PLE_ROW_WIDTH, 160);
        assert_eq!(QWEN4EXP_PLE_TOTAL_ROWS, 320_001_536);
        assert_eq!(QWEN4EXP_PLE_E4M3_BYTES, 51_200_245_760);
        assert_eq!(QWEN4EXP_PLE_DENSE_BF16_TENSOR_COUNT, 7);
        assert_eq!(QWEN4EXP_PLE_DENSE_BF16_ELEMENTS, 32_839_681);
        assert_eq!(QWEN4EXP_PLE_I64_TENSOR_COUNT, 3);
        assert_eq!(QWEN4EXP_PLE_I64_ELEMENTS, 35);
        assert_eq!(QWEN4EXP_PLE_TOTAL_TENSORS, 138);

        // The real table's 128 equal shards tile [0, 320001536) exactly, no gap or overlap, checked over
        // every real shard ordinal.
        let mut covered = 0usize;
        for shard in 0..QWEN4EXP_PLE_SHARDS {
            let range = qwen4exp_ple_shard_row_range(shard);
            assert_eq!(
                range.start, covered,
                "shard {shard} must start where the previous ended"
            );
            assert_eq!(range.len(), QWEN4EXP_PLE_ROWS_PER_SHARD);
            covered = range.end;
        }
        assert_eq!(covered, QWEN4EXP_PLE_TOTAL_ROWS);
    }
}

/// Card 449 H0b: structural rows for the whole-model exact packed tracers.
mod card449h0_exact_trace {
    use std::sync::Arc;

    use poot_graph_ir::op::PackedWeight;
    use poot_graph_ir::ops::PackedLinearGraphRow;
    use poot_quant::{OperandRole, PackedPayload, SourceRole};

    use super::*;

    fn descriptor(
        role: Qwen4ExpExpertProjection,
        hidden: usize,
        intermediate: usize,
    ) -> PackedWeight {
        let (out, k) = role.out_in(hidden, intermediate);
        PackedWeight::try_new(
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::Bf16,
            },
            [out, k],
        )
        .expect("bounded packed descriptor")
    }

    fn scale_elements(descriptor: PackedWeight) -> usize {
        let bytes_per_element = descriptor
            .format()
            .descriptor()
            .planar_operand(OperandRole::Scale)
            .expect("a packed weight has a Scale operand")
            .element_bytes();
        descriptor.source_bytes(SourceRole::Planar(OperandRole::Scale)) / bytes_per_element
    }

    fn packed_tables(
        layer: usize,
        experts: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Qwen4ExpPackedExpertTables {
        let tables = Qwen4ExpExpertProjection::ALL
            .into_iter()
            .map(|role| {
                let desc = descriptor(role, hidden, intermediate);
                let rows = (0..experts)
                    .map(|expert| {
                        let linear_id = role.checkpoint_prefix(layer, expert);
                        let weights =
                            vec![0x38u8; desc.source_bytes(SourceRole::Planar(OperandRole::Codes))];
                        let scales = (0..scale_elements(desc))
                            .flat_map(|_| 0x3f80u16.to_le_bytes())
                            .collect::<Vec<_>>();
                        let _owner = Arc::new(
                            PackedPayload::try_new(
                                desc,
                                [
                                    (SourceRole::Planar(OperandRole::Codes), weights.into()),
                                    (SourceRole::Planar(OperandRole::Scale), scales.into()),
                                ],
                            )
                            .expect("bounded packed payload"),
                        );
                        PackedLinearGraphRow {
                            ordinal: expert,
                            linear_id,
                            descriptor: desc,
                        }
                    })
                    .collect();
                Qwen4ExpPackedProjectionTable::new(role, rows)
            })
            .collect();
        Qwen4ExpPackedExpertTables::try_new(tables, layer, experts, hidden, intermediate)
            .expect("valid packed tables")
    }

    fn dense_roles(
        layer: usize,
        experts: usize,
        hidden: usize,
        shared: usize,
    ) -> [Qwen4ExpExactDenseRole; 5] {
        let name = |suffix: &str| format!("model.language_model.layers.{layer}.{suffix}");
        [
            Qwen4ExpExactDenseRole {
                name: name("mlp.gate.weight"),
                shape: [experts, hidden],
            },
            Qwen4ExpExactDenseRole {
                name: name("mlp.shared_expert.gate_proj.weight"),
                shape: [shared, hidden],
            },
            Qwen4ExpExactDenseRole {
                name: name("mlp.shared_expert.up_proj.weight"),
                shape: [shared, hidden],
            },
            Qwen4ExpExactDenseRole {
                name: name("mlp.shared_expert.down_proj.weight"),
                shape: [hidden, shared],
            },
            Qwen4ExpExactDenseRole {
                name: name("mlp.shared_expert_gate.weight"),
                shape: [1, hidden],
            },
        ]
    }

    fn tiny_exact_ple() -> Qwen4ExpPleConfig {
        Qwen4ExpPleConfig {
            ple_layer_ids: vec![2],
            ple_embed_dim: 8,
            ple_conv_kernel_size: 3,
            ngram_size: 3,
            ngram_vocab_size_base: 17,
            heads_per_ngram: 2,
            make_ngram_vocab_size_divisible_by: 8,
            seed: 1234,
            eos_token_id: 5,
            vocab_size: 11,
        }
    }

    fn exact_model_cfg() -> Qwen4ExpModelConfig {
        Qwen4ExpModelConfig {
            cfg: tiny_cfg(),
            qcfg: tiny_qcfg(4),
            gdn: Qwen4ExpGdnConfig {
                num_k_heads: 1,
                num_v_heads: 2,
                head_dim: 2,
                conv_k: 4,
            },
            layer_is_full: vec![false, false, false, true],
            vocab: 6,
            max_pos: 16,
            ffn_inter: 8,
            moe_n_experts: 3,
            moe_top_k: 2,
            moe_inter: 4,
            eps: 1e-5,
            chunk: 4,
            hc_count: 2,
            hc_lowrank: 3,
            ple: Some(tiny_exact_ple()),
        }
    }

    fn exact_sources(cfg: &Qwen4ExpModelConfig) -> Qwen4ExpExactPackedSources {
        let layers = cfg
            .layer_is_full
            .iter()
            .enumerate()
            .map(|(layer, _)| Qwen4ExpExactLayerSources {
                tables: packed_tables(layer, cfg.moe_n_experts, cfg.cfg.hidden, cfg.ffn_inter),
                dense: dense_roles(layer, cfg.moe_n_experts, cfg.cfg.hidden, cfg.ffn_inter),
            })
            .collect();
        let ple = cfg.ple.as_ref().expect("exact fixture has PLE");
        // Two bounded shards covering the padded tiny n-gram vocab so the sharded lookup has a
        // non-trivial owner split; row width is head_dim_per_ngram (each id indexes one head row).
        let _head_dim = ple.head_dim_per_ngram();
        let rows_per = 32usize;
        let ple_shards = (0..2)
            .map(|ordinal| Qwen4ExpPleShardRange {
                linear_id: format!(
                    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{ordinal}.weight"
                ),
                rows: (ordinal * rows_per)..((ordinal + 1) * rows_per),
            })
            .collect();
        Qwen4ExpExactPackedSources {
            layers,
            ple_shards,
            ple_weight_scale_name:
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
                    .to_string(),
        }
    }

    fn const_names(g: &Graph) -> Vec<String> {
        g.inputs
            .iter()
            .filter_map(|&id| g.meta(id).name.clone())
            .collect()
    }

    fn state_names(g: &Graph) -> Vec<String> {
        g.state
            .iter()
            .map(|(state_in, _)| {
                g.meta(*state_in)
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("v{state_in}"))
            })
            .collect()
    }

    #[test]
    fn card449h0_exact_prefill_validates_with_packed_ple_and_history_state() {
        let cfg = exact_model_cfg();
        let sources = exact_sources(&cfg);
        let graph = trace_qwen4exp_exact_prefill(&cfg, 4, 4, &sources)
            .expect("exact packed prefill traces");
        graph.validate().expect("exact packed prefill validates");
        let names = const_names(&graph);

        // Packed FFN: one weight+scale constant per expert projection, not the dense fused tensors.
        assert!(
            names
                .iter()
                .any(|n| n.contains("mlp.experts.0.gate_proj.packed_weight_source")),
            "packed gate weight must be declared: {names:?}"
        );
        assert!(
            !names
                .iter()
                .any(|n| n.ends_with("mlp.experts.gate_up_proj")),
            "dense fused expert tensor must not appear on the exact path"
        );
        // Routed dense roles are checkpoint-named BF16.
        assert!(
            names
                .iter()
                .any(|n| n == "model.language_model.layers.0.mlp.gate.weight")
        );
        // PLE: sharded E4M3 tables + weight_scale, no dense ngram_embedding.weight Const, and the
        // n-gram ids are composed in-graph (history State), not declared as a Const.
        assert!(
            names
                .iter()
                .any(|n| n.contains("ngram_embedding.shard_0.weight"))
        );
        assert!(
            names
                .iter()
                .any(|n| n.ends_with("ngram_embedding.weight_scale"))
        );
        assert!(
            !names.iter().any(|n| n.ends_with("ple.ngram_ids")),
            "option D composes n-gram ids in-graph; they must not be a Const"
        );
        assert!(
            !names
                .iter()
                .any(|n| n.ends_with("ple.ple_embedding.ngram_embedding.weight")),
            "exact PLE must not declare the dense n-gram table"
        );

        // History + PLE conv-cache sit at the PLE layer front, matching decode's layout so a
        // prefill output state feeds decode (Card 449 H0b).
        let states = state_names(&graph);
        let history = states
            .iter()
            .position(|n| n == "layers.1.ple.ngram_history")
            .expect("prefill carries the option-D history state");
        assert_eq!(
            graph.meta(graph.state[history].0).aval.dtype,
            DType::I32,
            "history state must be exact I32"
        );
        assert_eq!(
            graph.meta(graph.state[history].0).aval.shape,
            vec![1, tiny_exact_ple().context_len()]
        );
        let ple_conv = states
            .iter()
            .position(|n| n == "layers.1.ple.conv_cache")
            .expect("prefill carries the PLE conv cache so its state layout matches decode");
        assert!(
            history < ple_conv,
            "history is pushed before the PLE conv cache: {states:?}"
        );
        // QSA layer 3 carries k/v/idx like decode.
        let idx = states
            .iter()
            .position(|n| n == "layers.3.idx_k_cache")
            .expect("prefill carries the QSA idx_k_cache like decode");
        let k = states
            .iter()
            .position(|n| n == "layers.3.k_cache")
            .expect("prefill carries the QSA k_cache");
        assert!(k < idx, "k/v come before idx at the QSA layer: {states:?}");
        // Only layer 1 carries PLE state.
        assert!(
            !states
                .iter()
                .any(|n| n.starts_with("layers.0.ple.") || n.starts_with("layers.2.ple.")),
            "only the one-indexed PLE layer carries PLE state: {states:?}"
        );
    }

    #[test]
    fn card449h0_exact_decode_validates_with_history_and_conv_cache_state() {
        let cfg = exact_model_cfg();
        let sources = exact_sources(&cfg);
        let graph =
            trace_qwen4exp_exact_decode(&cfg, 4, &sources).expect("exact packed decode traces");
        graph.validate().expect("exact packed decode validates");
        let names = const_names(&graph);
        assert!(
            names
                .iter()
                .any(|n| n.contains("mlp.experts.0.gate_proj.packed_weight_source")),
            "decode uses the indexed packed FFN"
        );
        assert!(
            !names.iter().any(|n| n.ends_with("ple.ngram_ids")),
            "decode also composes n-gram ids in-graph"
        );
        let states = state_names(&graph);
        let history = states
            .iter()
            .position(|n| n == "layers.1.ple.ngram_history")
            .expect("decode carries the option-D history state");
        let conv = states
            .iter()
            .position(|n| n == "layers.1.ple.conv_cache")
            .expect("decode carries the PLE conv cache");
        assert_eq!(graph.meta(graph.state[history].0).aval.dtype, DType::I32);
        assert!(
            history < conv,
            "history is pushed before the conv cache at the PLE layer front: {states:?}"
        );
    }

    #[test]
    fn card449h0_exact_prefill_rejects_missing_ple_shards_and_wrong_layer_count() {
        let cfg = exact_model_cfg();
        let mut sources = exact_sources(&cfg);
        sources.ple_shards.clear();
        let error = trace_qwen4exp_exact_prefill(&cfg, 4, 4, &sources)
            .expect_err("PLE config without shard ranges must fail closed");
        assert!(
            matches!(error, Qwen4ExpExactTraceError::MissingPleShards),
            "unexpected error: {error}"
        );

        let mut sources = exact_sources(&cfg);
        sources.layers.pop();
        let error = trace_qwen4exp_exact_prefill(&cfg, 4, 4, &sources)
            .expect_err("short packed layer list must fail closed");
        assert!(
            matches!(
                error,
                Qwen4ExpExactTraceError::LayerCount {
                    expected: 4,
                    actual: 3
                }
            ),
            "unexpected error: {error}"
        );

        // Capacity below the real prompt length fails closed (Card 449 H0b).
        let sources = exact_sources(&cfg);
        let error = trace_qwen4exp_exact_prefill(&cfg, 4, 2, &sources)
            .expect_err("capacity below seq_len must fail closed");
        assert!(
            matches!(
                error,
                Qwen4ExpExactTraceError::CapacityBelowSeqLen {
                    seq_len: 4,
                    capacity: 2
                }
            ),
            "unexpected error: {error}"
        );
    }

    /// Card 449 H0b: prefill and decode graphs for the same capacity declare the same
    /// `graph.state` names in the same order with the same dtypes/shapes, so a prefill output
    /// state feeds decode directly (prefill-then-decode chaining).
    #[test]
    fn card449h0_prefill_and_decode_share_state_layout_for_chaining() {
        let cfg = exact_model_cfg();
        let sources = exact_sources(&cfg);
        let capacity = 8;
        let prefill = trace_qwen4exp_exact_prefill(&cfg, 4, capacity, &sources)
            .expect("exact packed prefill traces");
        let decode = trace_qwen4exp_exact_decode(&cfg, capacity, &sources)
            .expect("exact packed decode traces");
        let prefill_states = state_names(&prefill);
        let decode_states = state_names(&decode);
        assert_eq!(
            prefill_states, decode_states,
            "prefill and decode must declare identical state layouts for chaining"
        );
        assert_eq!(
            prefill.state.len(),
            decode.state.len(),
            "state pair counts must match"
        );
        for (i, ((pin, _), (din, _))) in prefill.state.iter().zip(&decode.state).enumerate() {
            let pin_av = &prefill.meta(*pin).aval;
            let din_av = &decode.meta(*din).aval;
            assert_eq!(
                pin_av.dtype, din_av.dtype,
                "state {i} dtype prefill vs decode"
            );
            let name = prefill
                .meta(*pin)
                .name
                .clone()
                .unwrap_or_else(|| format!("v{pin}"));
            // Match only the QSA cache names; `conv_cache` must not trip a `v_cache` substring.
            let is_kv = name.ends_with(".k_cache")
                || name.ends_with(".v_cache")
                || name.ends_with(".idx_k_cache");
            if is_kv {
                let heads = if name.ends_with(".idx_k_cache") {
                    1
                } else {
                    cfg.cfg.n_kv_heads
                };
                let dim = if name.ends_with(".idx_k_cache") {
                    cfg.qcfg.index_head_dim
                } else {
                    cfg.cfg.head_dim
                };
                assert_eq!(
                    pin_av.shape,
                    vec![1, heads, capacity, dim],
                    "KV/idx state {i} ({name}) must be capacity-sized"
                );
                assert_eq!(
                    din_av.shape, pin_av.shape,
                    "KV/idx state {i} ({name}) prefill vs decode shape"
                );
            } else {
                assert_eq!(
                    pin_av.shape, din_av.shape,
                    "non-KV state {i} ({name}) shape prefill vs decode"
                );
            }
        }
    }

    /// The index of the first equation that reads a graph value whose name satisfies `pred`, in
    /// program order. Constants and state inputs are both reachable this way, so it answers "at
    /// which point in the graph does this block start evaluating" rather than "in which order
    /// were the declarations created".
    fn first_eqn_using(g: &Graph, pred: impl Fn(&str) -> bool) -> Option<usize> {
        g.eqns.iter().position(|eqn| {
            eqn.inputs.iter().any(|operand| match operand {
                poot_graph_ir::Operand::Value(id) => g.meta(*id).name.as_deref().is_some_and(&pred),
                poot_graph_ir::Operand::Lit(_) => false,
            })
        })
    }

    /// Card 363 row `qwen4exp_ple_attaches_before_layer1_attention`: the exact whole-model graph
    /// places its PLE block only under zero-based decoder layer 1 (one-indexed `[2]`) and evaluates
    /// it BEFORE that layer's `attn_hyper_connection` - the front-of-layer injection point
    /// `trace.rs` documents and the row's own mutation ("interpret `[2]` as zero-based or insert
    /// after attention") targets. The "which layer" half is already pinned by
    /// `ple_layer_ids_are_one_indexed` and `trace_qwen38_prefill_places_ple_on_the_one_indexed_layer`;
    /// this test is the "before attention, in the exact graph" half, which nothing checked.
    #[test]
    fn qwen4exp_ple_attaches_before_layer1_attention() {
        let cfg = exact_model_cfg();
        let sources = exact_sources(&cfg);
        let graphs = [
            (
                "prefill",
                trace_qwen4exp_exact_prefill(&cfg, 4, 4, &sources).expect("exact prefill traces"),
            ),
            (
                "decode",
                trace_qwen4exp_exact_decode(&cfg, 4, &sources).expect("exact decode traces"),
            ),
        ];
        for (label, graph) in graphs {
            graph.validate().expect("exact graph validates");

            // PLE appears only at layer 1: every PLE-named constant and carried state is under
            // `layers.1.ple.`.
            for name in const_names(&graph).into_iter().chain(state_names(&graph)) {
                if name.contains(".ple.") {
                    assert!(
                        name.contains("layers.1.ple."),
                        "{label}: PLE value {name} must belong to zero-based decoder layer 1"
                    );
                }
            }

            let first_ple = first_eqn_using(&graph, |name| name.contains("layers.1.ple."))
                .expect("the exact graph evaluates a layer-1 PLE block");
            let first_hyper = first_eqn_using(&graph, |name| {
                name.starts_with("layers.1.attn_hyper_connection")
            })
            .expect("layer 1 has an attention hyper connection");
            assert!(
                first_ple < first_hyper,
                "{label}: PLE starts at equation {first_ple}, after layer 1's \
                 attn_hyper_connection at equation {first_hyper} - PLE must be injected at the \
                 front of the layer, before attention"
            );
            // The layer-1 mix itself (the GDN linear-attention block in this fixture; a full layer
            // would carry `self_attn.` instead), as a second, independent marker.
            if let Some(first_mix) = first_eqn_using(&graph, |name| {
                name.starts_with("layers.1.linear_attn.") || name.starts_with("layers.1.self_attn.")
            }) {
                assert!(
                    first_ple < first_mix,
                    "{label}: PLE starts at equation {first_ple}, after the layer-1 attention mix \
                     at equation {first_mix}"
                );
            }
        }
    }
}
