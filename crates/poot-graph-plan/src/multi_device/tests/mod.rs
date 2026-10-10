//! Card 360 acceptance/mutation tests, one file per row of the card's own acceptance table
//! Every row's fixture device count
//! and required mutation are named in that table; these tests honor it directly rather than a uniform
//! two-device default (see the card's own review note).

mod accounting;
mod capability;
mod communication_dependency;
mod model_neutral;
mod packed_residency;
mod placement_set;
mod replicated;
mod routed_exchange;
mod stage;

/// Small shared fixture builders reused by more than one row's test.
mod fixtures;
