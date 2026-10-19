use crate::builder::{Builder, Traced};
use crate::error::BuilderError;
use crate::op::BinOp;

/// Linear: matmul against a `[in, out]` weight, with an optional `[out]` bias broadcast-add. `compile`'s
/// contraction-epilogue fuse rule folds the add into the matmul kernel (Card 557); the trace states only
/// the primitives.
pub fn linear(
    b: &Builder,
    x: Traced,
    w: Traced,
    bias: Option<Traced>,
) -> Result<Traced, BuilderError> {
    let y = b.matmul(x, w)?;
    match bias {
        Some(bias) => b.binary(BinOp::Add, y, bias),
        None => Ok(y),
    }
}
