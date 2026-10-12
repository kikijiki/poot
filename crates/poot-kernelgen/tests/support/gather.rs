//! A gather of two rows of a four-row table, shared by the integration binaries that compile or dump a
//! representative movement body.

use poot_kernel_ir::{Body, Ty};
use poot_kernelgen::{MovementOp, MovementSpec, Storage, movement_body};

/// A gather of two rows of a four-row `[4, rest]` table in element type `dt`, with an `I32` index cast to
/// usize internally: [`movement_body`] over a fixed shape, for the tests that compile or
/// dump a representative body.
pub(crate) fn gather_axis0_dt(name: &str, dt: Ty, rest: usize) -> Body {
    movement_body(
        name,
        &MovementSpec::new(
            Storage::Elements { carrier: dt },
            vec![2, rest],
            MovementOp::Gather {
                in_shape: vec![4, rest],
                axis: 0,
                index_numel: 2,
            },
        ),
        None,
    )
    .expect("a two-row gather over a four-row table is a valid movement")
}
