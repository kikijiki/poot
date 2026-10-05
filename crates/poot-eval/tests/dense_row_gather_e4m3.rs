//! Card 405: coverage for the E4M3 row `DenseRowGather` CPU-oracle evaluation (`poot-eval/src/fp8.rs`,
//! `poot-eval/src/lib.rs`), alongside `DENSE_ROW_GATHER_SOURCE_DTYPES` (`poot-graph-ir/src/op.rs`) naming
//! `DType::E4M3FN` next to `DType::BF16`. See `specs/405-dense-row-gather-e4m3/spec.md`.
//!
//! Two hand-picked E4M3 byte values anchor the tests (hand-picked rather than generated so the data is
//! known to be non-degenerate):
//!
//! - byte `0x38` (sign 0, exponent `0111` = 7, mantissa `000`) decodes to exactly `1.0`: `2^(7-7) * 1`.
//! - byte `0x40` (sign 0, exponent `1000` = 8, mantissa `000`) decodes to exactly `2.0`: `2^(8-7) * 1`.

use poot_eval::fp8::e4m3fn_tensor;
use poot_eval::{EvalBudget, EvalOptions, Value};
use poot_graph_ir::{
    BinOp, DENSE_ROW_GATHER_SOURCE_DTYPES, Eqn, Graph, OpKind, Operand, Scalar, Slot, Storage,
    TensorType, ValueId, ValueMeta,
};
use poot_tensor::DType;
use poot_tensor::HostTensor;

use poot_quant::scalar::e4m3fn_to_f32;

/// Row 0's byte. Decodes to `1.0`.
const ROW0_BYTE: u8 = 0x38;
/// Row 1's byte. Decodes to `2.0`.
const ROW1_BYTE: u8 = 0x40;

/// A `[rows, row_width]` E4M3 table from raw bytes, row-major. Card 554d:
/// `fp8::the E4M3FN checkpoint-owner constructor` - the zero-copy checkpoint-owner constructor this
/// fixture used to build (a one-shard safetensors file, authenticated, classified
/// `TensorDisposition::StandaloneE4m3`, then loaded through the full `poot-load` pipeline just to get
/// an `Arc<ExactSourceOwner>` to wrap) - had no production caller anywhere in `poot-llm`/`poot-models`
/// and is deleted; this fixture now builds the carrier directly from the bytes it already has.
fn e4m3_table(shape: Vec<usize>, bytes: Vec<u8>) -> HostTensor {
    e4m3fn_tensor(shape, bytes).expect("fixture bytes match the declared shape")
}

/// Build a graph with one `[rows, row_width]` E4M3 `Const` table, one `[index_len]` I32 `Slot::Token` index,
/// and a `DenseRowGather { source: DType::E4M3FN }` reading it, plus, when `scale` is given, a downstream `Mul`
/// by that literal (PLE's composition shape: applied after the gather, not fused into it). `Builder` cannot
/// construct `DenseRowGather` (compiler-only, see `op.rs`), so this builds the `Graph` directly as
/// `poot-graph-plan`'s `fold_dense_bf16_row_gathers` pass does.
fn build_graph(
    rows: usize,
    row_width: usize,
    index_len: usize,
    scale: Option<f32>,
) -> (Graph, ValueId, ValueId) {
    let mut g = Graph::default();

    let table_shape = vec![rows, row_width];
    let table_id = g.values.len();
    g.values.push(ValueMeta::new(
        TensorType::new(table_shape.clone(), DType::E4M3FN),
        Storage::Const,
        Some("table".to_string()),
    ));
    g.inputs.push(table_id);
    g.consts.push(table_id);

    let index_shape = vec![index_len];
    let index_id = g.values.len();
    g.values.push(ValueMeta::new(
        TensorType::new(index_shape.clone(), DType::I32),
        Storage::Slot(Slot::Token),
        None,
    ));
    g.inputs.push(index_id);
    g.slots.push((index_id, Slot::Token));

    let gather_op = OpKind::DenseRowGather {
        source: DType::E4M3FN,
    };
    let gather_type = gather_op
        .infer(&[
            TensorType::new(table_shape, DType::E4M3FN),
            TensorType::new(index_shape, DType::I32),
        ])
        .expect("E4M3FN is an admitted DenseRowGather source (Card 405); operands are well-formed");
    let gather_id = g.values.len();
    g.values
        .push(ValueMeta::new(gather_type.clone(), Storage::Device, None));
    g.eqns.push(Eqn {
        op: gather_op,
        inputs: vec![Operand::Value(table_id), Operand::Value(index_id)],
        out: gather_id,
        layer: None,
    });

    let output_id = match scale {
        None => gather_id,
        Some(scale) => {
            let mul_op = OpKind::Binary(BinOp::Mul);
            let mul_type = mul_op
                .infer(&[gather_type, Scalar::F32(scale).ty()])
                .expect("F32 gather output times an F32 scalar literal broadcasts");
            let mul_id = g.values.len();
            g.values
                .push(ValueMeta::new(mul_type, Storage::Device, None));
            g.eqns.push(Eqn {
                op: mul_op,
                inputs: vec![Operand::Value(gather_id), Operand::Lit(Scalar::F32(scale))],
                out: mul_id,
                layer: None,
            });
            mul_id
        }
    };
    g.output = output_id;
    (g, table_id, index_id)
}

fn dense_tensor(value: Value) -> HostTensor {
    match value {
        Value::Host(t) => t,
        other => panic!("expected a dense F32 result, got {other:?}"),
    }
}

/// Card 405 SC-001/SC-002/SC-005 (mutation table rows 1 and 2). The two hand-picked rows decode to two
/// different non-zero values (printed and asserted distinct), and gathering each by its own index returns its
/// own value. It also runs `index = [1, 0]` and requires the reversed order, since a table that always read
/// row 0 (or the last row) would pass an `index = [0, 1]`-only check.
///
/// Required red mutation (row 1): read `table.bytes()[source_flat + 1]` instead of
/// `table.bytes()[source_flat]` in `fp8::dense_row_gather_e4m3`. Not yet run.
#[test]
fn dense_row_gather_e4m3_reads_owner_backed_rows() {
    let row0 = e4m3fn_to_f32(ROW0_BYTE);
    let row1 = e4m3fn_to_f32(ROW1_BYTE);
    eprintln!("dense_row_gather_e4m3_reads_owner_backed_rows: row0={row0} row1={row1}");
    assert_eq!(
        row0, 1.0,
        "byte 0x38 must decode to exactly 1.0, see module doc"
    );
    assert_eq!(
        row1, 2.0,
        "byte 0x40 must decode to exactly 2.0, see module doc"
    );
    assert_ne!(
        row0, row1,
        "SC-001/SC-005: the two fixture rows must be distinct before any mutation trusts them"
    );

    let table = e4m3_table(vec![2, 1], vec![ROW0_BYTE, ROW1_BYTE]);
    let (g, table_id, index_id) = build_graph(2, 1, 2, None);

    for index in [[0i32, 1], [1, 0]] {
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(table_id, Value::Host(table.clone()));
        inputs.insert(
            index_id,
            Value::Host(HostTensor::i32(vec![2], index.to_vec())),
        );
        let output = dense_tensor(
            poot_eval::eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
                .map(|r| r.output)
                .expect("gather-only graph evaluates"),
        );
        let expected = [
            e4m3fn_to_f32(if index[0] == 0 { ROW0_BYTE } else { ROW1_BYTE }),
            e4m3fn_to_f32(if index[1] == 0 { ROW0_BYTE } else { ROW1_BYTE }),
        ];
        eprintln!(
            "  index={index:?} -> output={:?} expected={expected:?}",
            output.as_f32().unwrap()
        );
        assert_eq!(output.as_f32().unwrap(), expected.as_slice());
    }
}

/// Card 405 FR-003 (mutation table row 3). Required red mutation: delete the bounds check inside
/// `fp8::dense_row_gather_e4m3` (the `find(...)` block) and feed the same out-of-range index; the gather would
/// read past `table.bytes()`, panicking or reading unrelated memory instead of naming the row. Not yet run.
#[test]
fn dense_row_gather_e4m3_still_checks_bounds() {
    let table = e4m3_table(vec![2, 1], vec![ROW0_BYTE, ROW1_BYTE]);
    let (g, table_id, index_id) = build_graph(2, 1, 1, None);

    let mut inputs = std::collections::HashMap::new();
    inputs.insert(table_id, Value::Host(table));
    // The table has 2 rows (valid indices 0..=1); 2 is out of range.
    inputs.insert(index_id, Value::Host(HostTensor::i32(vec![1], vec![2])));
    let result =
        poot_eval::eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).map(|r| r.output);
    // Card 555: the bounds check now routes through the one index rule's `EvalError::Index`,
    // not `Fp8Error::DenseRowGatherIndex`.
    let error = result.expect_err("expected a named out-of-range rejection");
    let poot_eval::EvalError::Index(fault) = &error else {
        panic!("expected EvalError::Index, got {error:?}")
    };
    assert_eq!(fault.value, poot_eval::IndexValue::I32(2));
    assert_eq!(fault.len, 2);
    assert_eq!(fault.kind, poot_eval::IndexFaultKind::OutOfRange);
}

/// Card 405 FR-006/SC-002/SC-003 (mutation table row 4). The same gather composed with a downstream `Mul` by a
/// non-trivial scale must match an independent `e4m3fn_to_f32(byte) * scale` reference that does not call the
/// production gather path.
///
/// Required red mutations, each run once and recorded (not asserted here, since they need production edits):
/// (a) mutate only the `Mul` scale literal in `build_graph` (e.g. `0.5` -> `0.75`): only this test's composed
///     assertion may change; `dense_row_gather_e4m3_reads_owner_backed_rows` must not, since it has no scale.
/// (b) mutate only the gather index (swap `ROW0_BYTE`/`ROW1_BYTE`, or the row the arm selects): both this
///     test and `dense_row_gather_e4m3_reads_owner_backed_rows` must change, since both read through the gather.
/// Not yet run.
#[test]
fn dense_row_gather_e4m3_composes_with_downstream_scale() {
    const SCALE: f32 = 0.5;
    let table = e4m3_table(vec![2, 1], vec![ROW0_BYTE, ROW1_BYTE]);
    let (g, table_id, index_id) = build_graph(2, 1, 2, Some(SCALE));

    let mut inputs = std::collections::HashMap::new();
    inputs.insert(table_id, Value::Host(table));
    inputs.insert(index_id, Value::Host(HostTensor::i32(vec![2], vec![0, 1])));
    let output = dense_tensor(
        poot_eval::eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| r.output)
            .expect("gather-then-scale graph evaluates"),
    );

    // Independent reference: e4m3fn_to_f32 is reused, but the multiply-by-scale is plain arithmetic, not a call
    // into `fp8::dense_row_gather_e4m3` or the Mul evaluator under test.
    let expected = [
        e4m3fn_to_f32(ROW0_BYTE) * SCALE,
        e4m3fn_to_f32(ROW1_BYTE) * SCALE,
    ];
    eprintln!(
        "dense_row_gather_e4m3_composes_with_downstream_scale: output={:?} expected={expected:?}",
        output.as_f32().unwrap()
    );
    assert_eq!(output.as_f32().unwrap(), expected.as_slice());
}

/// Checks that this file's premise (`DType::E4M3FN` is admitted) matches the graph-ir table, so a stale copy
/// of the dtype list here cannot diverge from `op.rs`.
#[test]
fn dense_row_gather_source_dtypes_include_e4m3fn_here_too() {
    assert!(DENSE_ROW_GATHER_SOURCE_DTYPES.contains(&DType::E4M3FN));
}
