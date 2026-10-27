//! POOT-1241 SC-001: `infer` is the one typing rule. Every operand of every op is substituted, one at a
//! time, with every other dtype, and the dtypes `infer` still admits there are exactly the ones the op's
//! definition names. An op that skipped a dtype relation (the six that did: `Concat`, `DynamicUpdateSlice`,
//! `ScatterUpdate`, `IndexedMatMul`, `Scatter`, `AllReduce`) admits a foreign dtype and its row goes red.

use poot_tensor::DType;

use crate::decompose::tests::{VARIANTS, canonical_rows, variant};
use crate::op::OpKind;
use crate::types::TensorType;

/// Every dtype a graph value can name.
const ALL: [DType; 15] = [
    DType::Bool,
    DType::I8,
    DType::U8,
    DType::I16,
    DType::U16,
    DType::I32,
    DType::U32,
    DType::I64,
    DType::U64,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
    DType::E4M3FN,
    DType::E8M0,
];

/// The dtypes other than `operands[index]`'s own that `op` still types over.
fn admitted_substitutes(op: &OpKind, operands: &[TensorType], index: usize) -> Vec<DType> {
    ALL.into_iter()
        .filter(|dtype| *dtype != operands[index].dtype)
        .filter(|dtype| {
            let mut mutated = operands.to_vec();
            mutated[index].dtype = *dtype;
            op.infer(&mutated).is_ok()
        })
        .collect()
}

/// What `infer` admits in place of one operand's dtype.
#[derive(Clone, Copy)]
enum Admits {
    /// No other dtype: the op names this operand's dtype, or ties it to another operand's.
    Nothing,
    /// The other half-width floats: the op is arithmetic over this operand and ties nothing else to it.
    OtherFloats,
    /// These dtypes only.
    Only(&'static [DType]),
    /// Every other dtype: the op moves this operand without reading a value (`DTypeClass` never applies).
    Anything,
}

impl Admits {
    fn dtypes(self, original: DType) -> Vec<DType> {
        let others = ALL.into_iter().filter(move |dtype| *dtype != original);
        match self {
            Admits::Nothing => Vec::new(),
            Admits::OtherFloats => others
                .filter(|dtype| matches!(dtype, DType::F16 | DType::BF16))
                .collect(),
            Admits::Only(dtypes) => dtypes.to_vec(),
            Admits::Anything => others.collect(),
        }
    }
}

/// The half-width storage dtypes an F32 value may be stored in and read back from (POOT-1395).
const NARROW_STORE: Admits = Admits::Only(&[DType::F16, DType::BF16]);

/// The expectation of every operand of every canonical row, by the row's name prefix. The names are checked
/// against the rows so the table cannot drift out of step with the fixture.
fn expectations() -> Vec<(&'static str, Vec<Admits>)> {
    use Admits::{Anything, Nothing, OtherFloats};
    let rigid = |operands: usize| vec![Nothing; operands];
    // POOT-1395: an F32 attention reads its keys and values from a half-width cache, widening on load.
    let narrow_kv = |operands: usize| {
        let mut admits = rigid(operands);
        admits[1] = NARROW_STORE;
        admits[2] = NARROW_STORE;
        admits
    };
    vec![
        ("neg", vec![OtherFloats]),
        ("add", rigid(2)),
        ("select", rigid(3)),
        ("select", rigid(3)),
        ("reduce", vec![OtherFloats]),
        ("broadcast", vec![Anything]),
        (
            "cast bf16",
            vec![Admits::Only(&[
                DType::I8,
                DType::I32,
                DType::F16,
                DType::BF16,
                DType::E4M3FN,
            ])],
        ),
        ("reshape", vec![Anything]),
        ("transpose", vec![Anything]),
        ("slice", vec![Anything]),
        ("concat", rigid(2)),
        ("iota", Vec::new()),
        ("gather", vec![Anything, Nothing]),
        // POOT-1395: a store narrows an F32 update into a half-width base (the KV cache's storage dtype).
        ("scatter_update", vec![NARROW_STORE, Nothing, Nothing]),
        // `A x B`: the left operand fixes the pairing, a half-width weight may follow an F32 activation.
        ("matmul", vec![Nothing, OtherFloats]),
        ("packed_dequant", rigid(2)),
        ("packed_contraction", rigid(3)),
        ("packed_contraction", rigid(3)),
        ("packed_row_gather", rigid(3)),
        ("dense_contraction bf16", rigid(2)),
        ("dense_contraction f32", rigid(2)),
        ("dense_contraction f16", rigid(2)),
        ("grouped_dense_contraction", rigid(2)),
        ("dense_row_gather bf16", rigid(2)),
        ("dense_row_gather e4m3fn", rigid(2)),
        (
            "contraction_epilogue",
            vec![Nothing, OtherFloats, Nothing, Nothing],
        ),
        ("arg_top_k", vec![OtherFloats]),
        ("expert_group", rigid(1)),
        ("indexed_matmul", vec![Nothing, OtherFloats, Nothing]),
        ("dyn_update_slice", vec![NARROW_STORE, Nothing, Nothing]),
        ("fused n_in=2", rigid(2)),
        ("fused n_in=1", rigid(1)),
        ("fused_row", vec![OtherFloats]),
        ("attention", narrow_kv(4)),
        ("attention", narrow_kv(4)),
        ("attention", narrow_kv(4)),
        ("attention", narrow_kv(5)),
        ("attention", narrow_kv(4)),
        ("attention", narrow_kv(5)),
        ("rope", rigid(3)),
        ("rope", rigid(3)),
        ("scan delta_rule", rigid(6)),
        ("scan ssd", rigid(7)),
        ("selected_attention", narrow_kv(5)),
        ("block_top_k", rigid(2)),
        ("all_reduce", vec![OtherFloats]),
        ("all_gather", vec![Anything]),
        ("random_uniform", rigid(1)),
        ("sample_token", rigid(1)),
    ]
}

#[test]
fn infer_admits_exactly_the_stated_dtypes_at_every_operand() {
    let rows = canonical_rows();
    let expected = expectations();
    assert_eq!(
        rows.len(),
        expected.len(),
        "one expectation per canonical row"
    );
    let mut covered = [false; VARIANTS];
    for (row, ((op, operands), (prefix, admits))) in rows.iter().zip(&expected).enumerate() {
        covered[variant(op)] = true;
        assert!(
            op.name().starts_with(prefix),
            "row {row} is {}, the table says {prefix}",
            op.name()
        );
        assert_eq!(operands.len(), admits.len(), "row {row} ({})", op.name());
        for (index, admit) in admits.iter().enumerate() {
            let want = admit.dtypes(operands[index].dtype);
            let got = admitted_substitutes(op, operands, index);
            assert_eq!(
                got,
                want,
                "{} operand {index} ({}): dtypes admitted in its place",
                op.name(),
                operands[index]
            );
        }
    }
    for (variant, covered) in covered.iter().enumerate() {
        assert!(*covered, "no canonical row exercises variant {variant}");
    }
}
