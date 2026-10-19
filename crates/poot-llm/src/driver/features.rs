//! Which features of a driver and of its requests compose.
//!
//! A [`Feature`] is one thing a driver configuration or a call can have: a draft policy, a placement,
//! recurrent state, an adapter, a beam search. The driver does not refuse a combination in the code
//! that meets it. It names the active features, as a [`FeatureSet`], and asks the one [`TABLE`] whether
//! they compose. A refusal is a typed [`Incompatible`] naming the pair and the reason the table gives.
//!
//! The table has a row for every feature, and a pair is refused by the row of the feature that comes
//! first in [`Feature::ALL`]. A test iterates every pair of features against the table and the driver
//! entry points, so a feature added to the list without a row, or a refusal without a driving
//! scenario, fails it.

use std::fmt;

/// Declares [`Feature`] and [`Feature::ALL`] from one list, so a feature cannot be in one and not the
/// other.
macro_rules! features {
    ($($(#[$meta:meta])* $name:ident => $label:literal,)+) => {
        /// One thing a driver configuration or a call can have, whose composition with the others the
        /// [`TABLE`] decides.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Feature {
            $($(#[$meta])* $name,)+
        }

        impl Feature {
            /// Every feature, in the order the table's rows follow.
            pub const ALL: &'static [Feature] = &[$(Feature::$name,)+];

            /// The feature as a refusal names it.
            pub const fn label(self) -> &'static str {
                match self {
                    $(Feature::$name => $label,)+
                }
            }
        }
    };
}

features! {
    /// Speculation by n-gram lookup in the sequence's own tokens.
    PromptLookup => "prompt-lookup speculation",
    /// Speculation by a second, smaller model.
    DraftModel => "draft-model speculation",
    /// Speculation by the model's own multi-token-prediction layer.
    Mtp => "MTP speculation",
    /// The program that proposes drafts (a draft model's or the MTP layer's) carries recurrent state.
    RecurrentDraftState => "recurrent state in the draft program",
    /// The served model carries recurrent state that folds every token.
    RecurrentState => "recurrent state",
    /// A caller-named verify window of drafts, beside the driver's own speculative rounds.
    VerifyWindows => "explicit verify windows",
    /// A request under an adapter-set generation.
    Adapters => "adapters",
    /// Beam search.
    Beam => "beam search",
    /// A request that supplies values for inputs its model declares.
    RequestInputs => "request inputs",
    /// Continuing an open sequence with the next turn's prompt.
    Resume => "resuming a sequence",
    /// An `encode` step beside the served executable.
    Encode => "encode steps",
    /// The paged KV pool.
    PagedKv => "the paged KV pool",
    /// A pipeline placement over several devices.
    PipelinePlacement => "a pipeline placement",
    /// A tensor-parallel placement over several devices.
    TensorPlacement => "a tensor placement",
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl Feature {
    const fn bit(self) -> u32 {
        1 << self as u32
    }
}

// A `FeatureSet` is one machine word.
const _: () = assert!(Feature::ALL.len() <= u32::BITS as usize);

/// The features a driver or a call has at once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FeatureSet(u32);

impl FeatureSet {
    /// No feature.
    pub const NONE: Self = Self(0);

    /// This set with `feature` added.
    #[must_use]
    pub const fn with(self, feature: Feature) -> Self {
        Self(self.0 | feature.bit())
    }

    /// This set with `feature` added when `present` holds.
    #[must_use]
    pub const fn with_if(self, present: bool, feature: Feature) -> Self {
        if present { self.with(feature) } else { self }
    }

    /// The features of both sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether `feature` is in the set.
    pub const fn contains(self, feature: Feature) -> bool {
        self.0 & feature.bit() != 0
    }

    /// Whether every feature of the set composes with the others: the first refusal the table holds
    /// against a pair inside it, in table order, otherwise `Ok`.
    pub fn check(self) -> Result<(), Incompatible> {
        for row in TABLE {
            if !self.contains(row.feature) {
                continue;
            }
            for &(other, reason) in row.refuses {
                if self.contains(other) {
                    return Err(Incompatible {
                        first: row.feature,
                        second: other,
                        reason,
                    });
                }
            }
        }
        Ok(())
    }
}

impl FromIterator<Feature> for FeatureSet {
    fn from_iter<I: IntoIterator<Item = Feature>>(features: I) -> Self {
        features
            .into_iter()
            .fold(Self::NONE, |set, feature| set.with(feature))
    }
}

/// Two features the driver cannot have at once, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{first} cannot be combined with {second}: {reason}")]
pub struct Incompatible {
    /// The feature whose table row refuses the pair.
    pub first: Feature,
    /// The feature that row names.
    pub second: Feature,
    /// What the table says about the pair.
    pub reason: &'static str,
}

/// One feature's refusals: each feature that comes later in [`Feature::ALL`] and cannot be combined with
/// it, with the reason.
pub(super) struct Row {
    /// The feature the row is for.
    pub(super) feature: Feature,
    /// The later features it cannot be combined with.
    pub(super) refuses: &'static [(Feature, &'static str)],
}

const ONE_DRAFT_POLICY: &str = "a driver has one draft policy";
const ONE_PLACEMENT: &str = "a driver has one placement";
const NO_DRAFT_PROGRAM: &str = "prompt lookup runs no draft program";
const DRAFT_PLACEMENT: &str =
    "the draft program's layers are not the target's, so the placement's cut names nothing in them";
const MTP_ROLLBACK: &str = "a rejected draft cannot be rolled back out of the MTP program's \
                            recurrent state, only the target's";
const MTP_BASE_WEIGHTS: &str = "the MTP layer drafts for the base weights, and an adapter changes \
                                the target's effective weights";
const MTP_BEAM_CACHE: &str = "beam search runs the target's programs without the hidden-state \
                              carry, so the MTP layer's cache would not follow its rows";
const TENSOR_ONE_EXECUTABLE: &str = "a tensor placement loads one executable whose weights are cut \
                                     per rank, so no second executable over whole weights is served";

/// The compatibility table: a row for every feature. A pair absent from every row composes.
pub(super) const TABLE: &[Row] = &[
    Row {
        feature: Feature::PromptLookup,
        refuses: &[
            (Feature::DraftModel, ONE_DRAFT_POLICY),
            (Feature::Mtp, ONE_DRAFT_POLICY),
            (Feature::RecurrentDraftState, NO_DRAFT_PROGRAM),
        ],
    },
    Row {
        feature: Feature::DraftModel,
        refuses: &[
            (Feature::Mtp, ONE_DRAFT_POLICY),
            (
                Feature::RecurrentDraftState,
                "a draft model's recurrent state cannot be rolled back, only the target's",
            ),
            (Feature::PipelinePlacement, DRAFT_PLACEMENT),
            (Feature::TensorPlacement, DRAFT_PLACEMENT),
        ],
    },
    Row {
        feature: Feature::Mtp,
        refuses: &[
            (Feature::RecurrentDraftState, MTP_ROLLBACK),
            (Feature::Adapters, MTP_BASE_WEIGHTS),
            (Feature::Beam, MTP_BEAM_CACHE),
            (
                Feature::PipelinePlacement,
                "the MTP layer's hidden-state carry and cache sit on one device past the target's stages",
            ),
            (
                Feature::TensorPlacement,
                "the MTP layer's hidden-state carry and cache sit on one device past the target's stages",
            ),
        ],
    },
    Row {
        feature: Feature::RecurrentDraftState,
        refuses: &[],
    },
    Row {
        feature: Feature::RecurrentState,
        refuses: &[(
            Feature::VerifyWindows,
            "only the driver's own speculative rounds snapshot recurrent state, so a rejected \
             draft of a named window would already have advanced it",
        )],
    },
    Row {
        feature: Feature::VerifyWindows,
        refuses: &[],
    },
    Row {
        feature: Feature::Adapters,
        refuses: &[(Feature::TensorPlacement, TENSOR_ONE_EXECUTABLE)],
    },
    Row {
        feature: Feature::Beam,
        refuses: &[],
    },
    Row {
        feature: Feature::RequestInputs,
        refuses: &[(
            Feature::Resume,
            "the sequence carries input values over its prompt that a new prompt does not",
        )],
    },
    Row {
        feature: Feature::Resume,
        refuses: &[(
            Feature::PagedKv,
            "a paged sequence shares earlier turns through the prefix cache: open the next turn instead",
        )],
    },
    Row {
        feature: Feature::Encode,
        refuses: &[(Feature::TensorPlacement, TENSOR_ONE_EXECUTABLE)],
    },
    Row {
        feature: Feature::PagedKv,
        refuses: &[],
    },
    Row {
        feature: Feature::PipelinePlacement,
        refuses: &[(Feature::TensorPlacement, ONE_PLACEMENT)],
    },
    Row {
        feature: Feature::TensorPlacement,
        refuses: &[],
    },
];

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// The table's verdict on the pair `a`, `b`: why they cannot be combined, `None` when they can.
    pub(in crate::driver) fn refusal(a: Feature, b: Feature) -> Option<Incompatible> {
        FeatureSet::NONE.with(a).with(b).check().err()
    }

    /// SC-001: the table has exactly one row for every feature, in the order of [`Feature::ALL`], and
    /// every refusal names a different, later feature, so a pair is decided by one row. Mutation: add a
    /// feature to the `features!` list without a row; this fails naming the feature.
    #[test]
    fn every_feature_has_one_row_and_every_pair_one_verdict() {
        let rows: Vec<Feature> = TABLE.iter().map(|row| row.feature).collect();
        assert_eq!(
            rows,
            Feature::ALL,
            "TABLE must have one row per feature, in the order of Feature::ALL"
        );
        for (at, row) in TABLE.iter().enumerate() {
            let mut seen = Vec::new();
            for &(other, reason) in row.refuses {
                assert!(
                    !reason.is_empty(),
                    "{:?} x {other:?} has no reason",
                    row.feature
                );
                let later = Feature::ALL[at + 1..].contains(&other);
                assert!(
                    later,
                    "{:?} x {other:?}: a row names only features that come later in Feature::ALL",
                    row.feature
                );
                assert!(
                    !seen.contains(&other),
                    "{:?} x {other:?} is refused twice",
                    row.feature
                );
                seen.push(other);
            }
        }
    }

    /// The verdict on a pair does not depend on the order it is named in, and a feature composes with
    /// itself.
    #[test]
    fn a_verdict_is_symmetric() {
        for &a in Feature::ALL {
            assert_eq!(refusal(a, a), None, "{a:?} with itself");
            for &b in Feature::ALL {
                assert_eq!(
                    refusal(a, b).map(|r| (r.reason, r.first.min(r.second), r.first.max(r.second))),
                    refusal(b, a).map(|r| (r.reason, r.first.min(r.second), r.first.max(r.second))),
                    "{a:?} x {b:?}"
                );
            }
        }
    }

    /// A set holding no refused pair passes, and one holding a refused pair names it.
    #[test]
    fn a_set_is_checked_pair_by_pair() {
        let ok: FeatureSet = [Feature::Mtp, Feature::PagedKv, Feature::RequestInputs]
            .into_iter()
            .collect();
        assert_eq!(ok.check(), Ok(()));
        let bad = ok.with(Feature::Adapters);
        let refused = bad.check().unwrap_err();
        assert_eq!(
            (refused.first, refused.second),
            (Feature::Mtp, Feature::Adapters)
        );
    }
}
