//! Card 1011: `Cast(BF16 -> F32)` of a BF16 const, which every backend plans as the imported
//! `packed_bf16_to_f32` body over the const's packed `u32` lanes. The cases sit where that body can go
//! wrong: an odd element count (the last lane is half used, so the `i / 2 < words.len()` tail guard and the
//! shift of the high half both matter) and words a widening must carry bit for bit (a quiet NaN with a payload,
//! a negative one, a subnormal, a negative subnormal).

use std::sync::Arc;

use poot_graph_ir::{Builder, TensorType};
use poot_graph_plan::{FusionPolicy, ImportedKernel, KernelChoice, Submission, Target};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::{DType, HostTensor};

use poot_test_util::StepFixture;

use crate::Fixture;

/// +NaN with payload bits set.
const NAN_PAYLOAD: u16 = 0x7fc1;
/// -NaN with payload bits set.
const NEG_NAN_PAYLOAD: u16 = 0xffc1;
/// The smallest positive subnormal.
const SUBNORMAL: u16 = 0x0001;
/// The smallest negative subnormal.
const NEG_SUBNORMAL: u16 = 0x8001;
/// 1.0.
const ONE: u16 = 0x3f80;

/// The cases: `(name, stored BF16 words)`. Five elements end in a half-used lane; each single-element case
/// is a lone low half.
const CASES: [(&str, &[u16]); 4] = [
    (
        "bf16_cast_five_specials",
        &[NAN_PAYLOAD, NEG_NAN_PAYLOAD, SUBNORMAL, NEG_SUBNORMAL, ONE],
    ),
    ("bf16_cast_one_nan_payload", &[NAN_PAYLOAD]),
    ("bf16_cast_one_negative_nan_payload", &[NEG_NAN_PAYLOAD]),
    ("bf16_cast_one_subnormal", &[SUBNORMAL]),
];

/// One fixture per case: a BF16 const `x` of the case's words, cast to F32 and returned, with a single
/// (slot-free) step.
pub fn bf16_const_cast_fixtures() -> Vec<Fixture> {
    CASES
        .iter()
        .map(|&(name, words)| {
            let b = Builder::new();
            let x = b.constant("x", TensorType::new(vec![words.len()], DType::BF16));
            let widened = b.cast(x, DType::F32);
            let graph = b.finish(widened);
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let mut store = WeightStore::builder();
            store
                .insert(
                    "x",
                    WeightEntry::Dense(
                        DenseWeight::try_new(DType::BF16, vec![words.len()], Arc::from(bytes))
                            .expect("a BF16 dense weight"),
                    ),
                )
                .expect("one weight");
            Fixture {
                name,
                graph,
                store: store.build(),
                steps: vec![Vec::<StepFixture>::new()],
                fusion: FusionPolicy::Full,
            }
        })
        .collect()
}

/// Every case's device output equals both the oracle's and the widening the BF16 format defines (the word
/// shifted into the high half of an F32), bit for bit. The second comparison is independent of the oracle, so a
/// shared fault in how both read a NaN payload cannot hide.
pub fn assert_bit_exact(fixture: &Fixture, got: &HostTensor, oracle: &HostTensor) {
    let (_, words) = CASES
        .iter()
        .find(|(name, _)| *name == fixture.name)
        .expect("a fixture from bf16_const_cast_fixtures");
    let bits = |t: &HostTensor| -> Vec<u32> {
        t.as_f32()
            .expect("a widened BF16 is F32")
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    let want: Vec<u32> = words.iter().map(|&w| u32::from(w) << 16).collect();
    assert_eq!(bits(oracle), want, "{}: oracle words", fixture.name);
    assert_eq!(bits(got), want, "{}: device words", fixture.name);
}

/// `fixture`'s program, compiled for `target`, runs its one `Cast` through the imported `packed_bf16_to_f32`
/// body, so a row shows the kernel it ran is the packed reader and not only that the words matched.
pub fn assert_one_packed_cast(fixture: &Fixture, target: Target) {
    let staged = crate::staged(fixture, target, Submission::Replay);
    let kernels: Vec<ImportedKernel> = staged
        .stages()
        .flat_map(|(_, _, program)| {
            program.planned().filter_map(move |(eqn, _)| {
                match (
                    eqn.op.name().starts_with("cast"),
                    program.kernel_choice(eqn),
                ) {
                    (true, KernelChoice::Imported { kernel, .. }) => Some(*kernel),
                    _ => None,
                }
            })
        })
        .collect();
    assert_eq!(
        kernels,
        [ImportedKernel::PackedBf16ToF32],
        "{}: the cast runs the packed reader",
        fixture.name
    );
}
