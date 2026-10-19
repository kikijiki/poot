//! SC-001: the compatibility table against the driver's own entry points.
//!
//! [`features::tests`](crate::driver::features) pins the table's shape: a row for every feature and one
//! verdict for every pair. This iterates every pair of features and checks the verdict where it is
//! observable: every pair the table refuses is driven through the real entry point that meets it (a
//! constructor, `prepare`, `open`, `resume`, `beam`, `prepare_encode`), over inputs where the two sides
//! differ, and must come back as the typed refusal naming that pair. A pair the table refuses that has no
//! scenario fails, so a new refusal row cannot land without one. A pair no driver can hold at once (one
//! draft policy, one placement) is refused by the table alone, and is listed as such.

use crate::driver::beam::BeamError;
use crate::driver::error::{DriverError, Unsupported};
use crate::driver::features::tests::refusal;
use crate::driver::placed_tests::{
    witness_draft_model_placement, witness_mtp_placement, witness_tensor_adapters,
    witness_tensor_encode,
};
use crate::driver::step::mtp_tests::{
    witness_mtp_adapters, witness_mtp_beam, witness_mtp_recurrent_draft_state,
};
use crate::driver::{Feature, Incompatible};

use super::serving_tests::{witness_recurrent_verify_windows, witness_resume_with_inputs};
use super::{witness_recurrent_draft_model, witness_resume_paged};

/// How a refused pair is observed.
enum Observed {
    /// The error the driver returned when the pair met at its entry point.
    Driven(DriverError),
    /// No driver holds both at once: the options take one value of each kind.
    OneOfAKind,
    /// No scenario drives the pair.
    Undriven,
}

fn observe(first: Feature, second: Feature) -> Observed {
    use Feature::{
        Adapters, Beam, DraftModel, Encode, Mtp, PagedKv, PipelinePlacement, PromptLookup,
        RecurrentDraftState, RecurrentState, RequestInputs, Resume, TensorPlacement, VerifyWindows,
    };
    let driven = Observed::Driven;
    match (first, second) {
        (PromptLookup, DraftModel | Mtp | RecurrentDraftState)
        | (DraftModel, Mtp)
        | (PipelinePlacement, TensorPlacement) => Observed::OneOfAKind,
        (DraftModel, RecurrentDraftState) => driven(witness_recurrent_draft_model()),
        (DraftModel, placed @ (PipelinePlacement | TensorPlacement)) => {
            driven(witness_draft_model_placement(placed))
        }
        (Mtp, RecurrentDraftState) => driven(witness_mtp_recurrent_draft_state()),
        (Mtp, Adapters) => driven(witness_mtp_adapters()),
        (Mtp, Beam) => match witness_mtp_beam() {
            BeamError::Driver(error) => driven(error),
            other => panic!("beam under MTP failed some other way: {other:?}"),
        },
        (Mtp, placed @ (PipelinePlacement | TensorPlacement)) => {
            driven(witness_mtp_placement(placed))
        }
        (RecurrentState, VerifyWindows) => driven(witness_recurrent_verify_windows()),
        (Adapters, TensorPlacement) => driven(witness_tensor_adapters()),
        (RequestInputs, Resume) => driven(witness_resume_with_inputs()),
        (Resume, PagedKv) => driven(witness_resume_paged()),
        (Encode, TensorPlacement) => driven(witness_tensor_encode()),
        _ => Observed::Undriven,
    }
}

/// Every pair of features: the driver refuses it exactly when the table does, and names it. Mutation: drop
/// the `(Feature::Adapters, ..)` refusal of the `Mtp` row in `features::TABLE`; the MTP adapter request
/// opens past the compose and fails some other way, and this fails naming the pair. Add a refusal row
/// with no scenario; this fails naming it.
#[test]
fn the_driver_refuses_exactly_the_pairs_the_table_refuses() {
    let all = Feature::ALL;
    let mut driven = 0;
    for (at, &first) in all.iter().enumerate() {
        for &second in &all[at + 1..] {
            match (refusal(first, second), observe(first, second)) {
                (None, Observed::Undriven) => {}
                (None, _) => panic!(
                    "{first:?} x {second:?}: the table composes the pair but its scenario fails; \
                     the scenario or the table is wrong"
                ),
                (Some(_), Observed::Undriven) => panic!(
                    "{first:?} x {second:?}: the table refuses the pair and no scenario drives it; \
                     add one to `observe`"
                ),
                (Some(_), Observed::OneOfAKind) => {}
                (Some(table), Observed::Driven(error)) => {
                    driven += 1;
                    let DriverError::Unsupported(Unsupported::Incompatible(Incompatible {
                        first: got_first,
                        second: got_second,
                        reason,
                    })) = error
                    else {
                        panic!("{first:?} x {second:?}: the driver returned {error:?}");
                    };
                    assert_eq!(
                        (got_first, got_second, reason),
                        (first, second, table.reason),
                        "{first:?} x {second:?}"
                    );
                }
            }
        }
    }
    assert!(driven >= 13, "{driven} refused pairs were driven");
}
