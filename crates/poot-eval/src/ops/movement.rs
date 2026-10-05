//! Movement: `Reshape`, `Transpose`, `Slice`, `Concat`, `Broadcast`, `DynamicUpdateSlice`,
//! `ScatterUpdate` and axis-0 `Scatter`, over the host carrier and the storage-aware [`Value`] lane.
//!
//! [`generic`] holds one function per op, over `T: Copy`: `Transpose`/`Slice`/`Concat`/`Broadcast`
//! are pure geometry with no runtime index to fault on; `DynamicUpdateSlice`'s runtime start,
//! `ScatterUpdate`'s inverse (its `-1` "keep base" sentinel handled before any other value reaches
//! [`index_at`](super::index_rule::index_at)) and axis-0 `Scatter`'s index all route through the one
//! index rule shared with every other index-consuming op (R470-011). The rest of this module is the
//! thin per-class plumbing: a [`HostTensor`]'s payload is one of four storage classes (`f32`, `i32`,
//! half-width words, raw bytes), each op runs [`generic`] over that class and republishes the same
//! dtype, so a BF16/F16/E4M3FN operand moves its own words and bytes and is never widened to f32.

#[cfg(any(test, feature = "test-support"))]
use poot_graph_ir::OpKind;
use poot_graph_ir::ValueId;
use poot_tensor::{HostData, HostTensor};

use super::index_rule::IndexValue;
use crate::{EvalError, PackedEvalError, Value};

/// An index-role operand's values, typed ([`IndexValue`]). An index-role operand is declared F32 or
/// I32 ([`poot_tensor::DType::is_index_operand`]); any other dtype is refused.
pub(crate) fn index_values(t: &HostTensor) -> Result<Vec<IndexValue>, EvalError> {
    match t.data() {
        HostData::F32(values) => Ok(values.iter().map(|&v| IndexValue::F32(v)).collect()),
        HostData::I32(words) => Ok(words.iter().map(|&v| IndexValue::I32(v)).collect()),
        _ => Err(EvalError::unsupported(
            "index",
            format!("an index operand is F32 or I32, got {}", t.dtype()),
        )),
    }
}

/// The generic engine: pure row-major geometry over `T: Copy`, shared by the dense f32/i32 lanes and
/// E4M3FN's raw bytes. No runtime shape validation here: `infer` already restricts every static shape
/// (`perm`, `axis`, `start`/`end`, the parts' shapes) to one this geometry always indexes in range.
pub(crate) mod generic {
    use poot_graph_ir::ValueId;

    use super::super::index_rule::{IndexValue, index_at};
    use crate::{EvalError, broadcast_flat, strides, unravel};

    pub(crate) fn transpose<T: Copy>(data: &[T], shape: &[usize], perm: &[usize]) -> Vec<T> {
        let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let in_st = strides(shape);
        let n: usize = out_shape.iter().product();
        (0..n)
            .map(|flat| {
                let out_idx = unravel(flat, &out_shape);
                let src: usize = out_idx
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| c * in_st[perm[i]])
                    .sum();
                data[src]
            })
            .collect()
    }

    pub(crate) fn slice<T: Copy>(
        data: &[T],
        shape: &[usize],
        axis: usize,
        start: usize,
        end: usize,
    ) -> Vec<T> {
        let mut out_shape = shape.to_vec();
        out_shape[axis] = end - start;
        let in_st = strides(shape);
        let n: usize = out_shape.iter().product();
        (0..n)
            .map(|flat| {
                let mut idx = unravel(flat, &out_shape);
                idx[axis] += start;
                let src: usize = idx.iter().zip(&in_st).map(|(c, st)| c * st).sum();
                data[src]
            })
            .collect()
    }

    pub(crate) fn concat<T: Copy>(parts: &[(&[T], &[usize])], axis: usize) -> Vec<T> {
        let mut out_shape = parts[0].1.to_vec();
        out_shape[axis] = parts.iter().map(|(_, shape)| shape[axis]).sum();
        let n: usize = out_shape.iter().product();
        (0..n)
            .map(|flat| {
                let mut idx = unravel(flat, &out_shape);
                let mut remaining = idx[axis];
                let (data, shape) = parts
                    .iter()
                    .find(|(_, shape)| {
                        if remaining < shape[axis] {
                            true
                        } else {
                            remaining -= shape[axis];
                            false
                        }
                    })
                    .expect("out_shape's axis extent is the sum of every part's");
                idx[axis] = remaining;
                let st = strides(shape);
                let src: usize = idx.iter().zip(&st).map(|(c, s)| c * s).sum();
                data[src]
            })
            .collect()
    }

    pub(crate) fn broadcast<T: Copy>(
        data: &[T],
        shape: &[usize],
        target_shape: &[usize],
    ) -> Vec<T> {
        let n: usize = target_shape.iter().product();
        (0..n)
            .map(|flat| data[broadcast_flat(&unravel(flat, target_shape), shape)])
            .collect()
    }

    /// Write `update` into a copy of `operand` starting at `start` along `axis`; `start` is already
    /// validated against the window (every caller resolves it through [`index_at`] first).
    pub(crate) fn dynamic_update_slice<T: Copy>(
        operand: &[T],
        operand_shape: &[usize],
        update: &[T],
        update_shape: &[usize],
        start: usize,
        axis: usize,
    ) -> Vec<T> {
        let mut out = operand.to_vec();
        let op_st = strides(operand_shape);
        for (flat, &value) in update.iter().enumerate() {
            let mut idx = unravel(flat, update_shape);
            idx[axis] += start;
            let dst: usize = idx.iter().zip(&op_st).map(|(c, st)| c * st).sum();
            out[dst] = value;
        }
        out
    }

    /// `out[p] = inverse[p] >= 0 ? src[inverse[p]] : base[p]` over axis-0 rows (the eager oracle for
    /// the GPU `scatter_update_dt`). `inverse`'s `-1` sentinel ("keep base") is handled here, before
    /// any other value reaches [`index_at`]: every other negative value, and `src`'s own row count,
    /// fault through it.
    pub(crate) fn scatter_update<T: Copy>(
        base: &[T],
        base_shape: &[usize],
        src: &[T],
        src_rows: usize,
        inverse: &[IndexValue],
        eqn: ValueId,
    ) -> Result<Vec<T>, EvalError> {
        let pool = base_shape[0];
        let rest: usize = base_shape[1..].iter().product::<usize>().max(1);
        let mut out = base.to_vec();
        for (p, &value) in inverse.iter().enumerate().take(pool) {
            let keep_base = matches!(value, IndexValue::F32(v) if v == -1.0)
                || matches!(value, IndexValue::I32(-1));
            if keep_base {
                continue;
            }
            let row = index_at(value, p, src_rows, eqn)?;
            let (dst, src_at) = (p * rest, row * rest);
            out[dst..dst + rest].copy_from_slice(&src[src_at..src_at + rest]);
        }
        Ok(out)
    }

    /// Scatter rows of `src` along axis 0 (the inverse of axis-0 [`super::super::gather::gather`]):
    /// `out[index[j], ..] = src[j, ..]`. `index` is a `[N]` permutation of `0..N` (N =
    /// `src_shape[0]`); `out` has the same shape as `src`. Rows no index targets stay `T::default()`
    /// (a permutation writes all of them).
    pub(crate) fn scatter_axis0<T: Copy + Default>(
        src: &[T],
        src_shape: &[usize],
        index: &[IndexValue],
        eqn: ValueId,
    ) -> Result<Vec<T>, EvalError> {
        let n = src_shape[0];
        let rest: usize = src_shape[1..].iter().product::<usize>().max(1);
        let mut out = vec![T::default(); src.len()];
        for (j, &value) in index.iter().enumerate().take(n) {
            let dst = index_at(value, j, n, eqn)?;
            let (dst_at, src_at) = (dst * rest, j * rest);
            out[dst_at..dst_at + rest].copy_from_slice(&src[src_at..src_at + rest]);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// The host lane: one dispatch over the four storage classes, republishing the operand's own dtype.
// ---------------------------------------------------------------------------------------------

/// Run `$body` (an expression over `$d: &[T]`) on the payload of `$x` in its own storage class and
/// republish the result as the same class.
macro_rules! in_class {
    ($x:expr, $d:ident => $body:expr) => {
        match $x.data() {
            HostData::F32($d) => {
                let $d: &[f32] = $d;
                HostData::F32($body.into())
            }
            HostData::I32($d) => {
                let $d: &[i32] = $d;
                HostData::I32($body.into())
            }
            HostData::Half($d) => {
                let $d: &[u16] = $d;
                HostData::Half($body.into())
            }
            HostData::Bytes($d) => {
                let $d: &[u8] = $d;
                HostData::Bytes($body.into())
            }
        }
    };
}

fn republish(x: &HostTensor, shape: Vec<usize>, data: HostData) -> Result<HostTensor, EvalError> {
    Ok(HostTensor::new(x.dtype(), shape, data)?)
}

/// One part of a concat: its `T` lane and its shape.
type LanePart<'a, T> = (&'a [T], &'a [usize]);

/// The parts' payloads as one `T` lane each, or a refusal when the parts disagree on dtype.
fn lane_parts<'a, T>(
    parts: &[&'a HostTensor],
    lane: impl Fn(&'a HostData) -> Option<&'a [T]>,
) -> Result<Vec<LanePart<'a, T>>, EvalError> {
    parts
        .iter()
        .map(|part| {
            lane(part.data())
                .map(|data| (data, part.shape()))
                .ok_or_else(|| {
                    EvalError::unsupported("concat", "Concat parts must all share one dtype")
                })
        })
        .collect()
}

fn f32_lane(data: &HostData) -> Option<&[f32]> {
    match data {
        HostData::F32(v) => Some(v),
        _ => None,
    }
}

fn i32_lane(data: &HostData) -> Option<&[i32]> {
    match data {
        HostData::I32(v) => Some(v),
        _ => None,
    }
}

fn half_lane(data: &HostData) -> Option<&[u16]> {
    match data {
        HostData::Half(v) => Some(v),
        _ => None,
    }
}

fn byte_lane(data: &HostData) -> Option<&[u8]> {
    match data {
        HostData::Bytes(v) => Some(v),
        _ => None,
    }
}

/// `Reshape`: a metadata-only view sharing the payload allocation; the element count is rechecked.
pub(crate) fn reshape(x: &HostTensor, shape: &[usize]) -> Result<HostTensor, EvalError> {
    Ok(x.reshaped(shape.to_vec())?)
}

pub(crate) fn broadcast_to(x: &HostTensor, shape: &[usize]) -> Result<HostTensor, EvalError> {
    let data = in_class!(x, d => generic::broadcast(d, x.shape(), shape));
    republish(x, shape.to_vec(), data)
}

pub(crate) fn transpose(x: &HostTensor, perm: &[usize]) -> Result<HostTensor, EvalError> {
    let out_shape: Vec<usize> = perm.iter().map(|&p| x.shape()[p]).collect();
    let data = in_class!(x, d => generic::transpose(d, x.shape(), perm));
    republish(x, out_shape, data)
}

pub(crate) fn slice(
    x: &HostTensor,
    axis: usize,
    start: usize,
    end: usize,
) -> Result<HostTensor, EvalError> {
    let mut out_shape = x.shape().to_vec();
    out_shape[axis] = end - start;
    let data = in_class!(x, d => generic::slice(d, x.shape(), axis, start, end));
    republish(x, out_shape, data)
}

pub(crate) fn concat(parts: &[&HostTensor], axis: usize) -> Result<HostTensor, EvalError> {
    let mut out_shape = parts[0].shape().to_vec();
    out_shape[axis] = parts.iter().try_fold(0usize, |extent, part| {
        extent
            .checked_add(part.shape()[axis])
            .ok_or_else(|| EvalError::unsupported("concat", "Concat axis extent overflows usize"))
    })?;
    if parts.iter().any(|part| part.dtype() != parts[0].dtype()) {
        return Err(EvalError::unsupported(
            "concat",
            "Concat parts must all share one dtype",
        ));
    }
    let data = match parts[0].data() {
        HostData::F32(_) => {
            HostData::F32(generic::concat(&lane_parts(parts, f32_lane)?, axis).into())
        }
        HostData::I32(_) => {
            HostData::I32(generic::concat(&lane_parts(parts, i32_lane)?, axis).into())
        }
        HostData::Half(_) => {
            HostData::Half(generic::concat(&lane_parts(parts, half_lane)?, axis).into())
        }
        HostData::Bytes(_) => {
            HostData::Bytes(generic::concat(&lane_parts(parts, byte_lane)?, axis).into())
        }
    };
    republish(parts[0], out_shape, data)
}

/// Scatter rows of `src` along axis 0; `index` is `src`'s own index-role operand values. An index
/// that does not select a row of `0..N` is [`EvalError::Index`], naming `eqn`.
pub(crate) fn scatter_axis0(
    src: &HostTensor,
    index: &HostTensor,
    eqn: ValueId,
) -> Result<HostTensor, EvalError> {
    let index_values = index_values(index)?;
    let data = match src.data() {
        HostData::F32(d) => {
            HostData::F32(generic::scatter_axis0(d, src.shape(), &index_values, eqn)?.into())
        }
        HostData::I32(d) => {
            HostData::I32(generic::scatter_axis0(d, src.shape(), &index_values, eqn)?.into())
        }
        HostData::Half(d) => {
            HostData::Half(generic::scatter_axis0(d, src.shape(), &index_values, eqn)?.into())
        }
        HostData::Bytes(d) => {
            HostData::Bytes(generic::scatter_axis0(d, src.shape(), &index_values, eqn)?.into())
        }
    };
    republish(src, src.shape().to_vec(), data)
}

pub(crate) fn scatter_update(
    base: &HostTensor,
    src: &HostTensor,
    inverse: &HostTensor,
    out_shape: &[usize],
    eqn: ValueId,
) -> Result<HostTensor, EvalError> {
    let inverse_values = index_values(inverse)?;
    let src_rows = src.shape().first().copied().unwrap_or(0);
    let data = match (base.data(), src.data()) {
        (HostData::F32(b), HostData::F32(s)) => HostData::F32(
            generic::scatter_update(b, base.shape(), s, src_rows, &inverse_values, eqn)?.into(),
        ),
        (HostData::I32(b), HostData::I32(s)) => HostData::I32(
            generic::scatter_update(b, base.shape(), s, src_rows, &inverse_values, eqn)?.into(),
        ),
        (HostData::Half(b), HostData::Half(s)) => HostData::Half(
            generic::scatter_update(b, base.shape(), s, src_rows, &inverse_values, eqn)?.into(),
        ),
        (HostData::Bytes(b), HostData::Bytes(s)) => HostData::Bytes(
            generic::scatter_update(b, base.shape(), s, src_rows, &inverse_values, eqn)?.into(),
        ),
        _ => {
            return Err(EvalError::unsupported(
                "scatter_update",
                "ScatterUpdate base and src must share one dtype",
            ));
        }
    };
    republish(base, out_shape.to_vec(), data)
}

/// Write `update` into a copy of `operand` starting at `index` along `axis` (XLA DynamicUpdateSlice,
/// single axis). The KV-cache slot write. `index` is already validated against the window.
pub(crate) fn dynamic_update_slice(
    operand: &HostTensor,
    update: &HostTensor,
    index: usize,
    axis: usize,
) -> Result<HostTensor, EvalError> {
    let data = match (operand.data(), update.data()) {
        (HostData::F32(o), HostData::F32(u)) => HostData::F32(
            generic::dynamic_update_slice(o, operand.shape(), u, update.shape(), index, axis)
                .into(),
        ),
        (HostData::I32(o), HostData::I32(u)) => HostData::I32(
            generic::dynamic_update_slice(o, operand.shape(), u, update.shape(), index, axis)
                .into(),
        ),
        (HostData::Half(o), HostData::Half(u)) => HostData::Half(
            generic::dynamic_update_slice(o, operand.shape(), u, update.shape(), index, axis)
                .into(),
        ),
        (HostData::Bytes(o), HostData::Bytes(u)) => HostData::Bytes(
            generic::dynamic_update_slice(o, operand.shape(), u, update.shape(), index, axis)
                .into(),
        ),
        _ => {
            return Err(EvalError::unsupported(
                "dynamic_update_slice",
                "DynamicUpdateSlice operand and update must share one dtype",
            ));
        }
    };
    republish(operand, operand.shape().to_vec(), data)
}

/// One movement equation on the host, independent of the walk's own per-equation loop. `eqn` is
/// `0` (no equation id in scope at this depth, the `EvalError::unsupported` convention).
///
/// Card 626: this card's own executor-fallback callers (the pre-contract device executors) died in
/// Cards 546b/549, so production no longer reaches this; it stays `#[cfg(any(test, feature =
/// "test-support"))]` for `poot-graph-plan`'s strided-view fuzz test (spec 132), which checks a view's
/// index math against this oracle - `poot-graph-plan`'s dev-dependency on this crate enables
/// `test-support`.
#[cfg(any(test, feature = "test-support"))]
pub fn apply_movement(op: &OpKind, ins: &[HostTensor]) -> Result<HostTensor, EvalError> {
    const NO_EQN: ValueId = 0;
    match op {
        OpKind::Reshape { shape } => reshape(&ins[0], shape),
        OpKind::Transpose { perm } => transpose(&ins[0], perm),
        OpKind::Slice { axis, start, end } => slice(&ins[0], *axis, *start, *end),
        OpKind::Concat { axis } => {
            let refs: Vec<&HostTensor> = ins.iter().collect();
            concat(&refs, *axis)
        }
        OpKind::Gather { axis } => {
            let index_values = index_values(&ins[1])?;
            let mut out_shape = ins[0].shape()[..*axis].to_vec();
            out_shape.extend_from_slice(ins[1].shape());
            out_shape.extend_from_slice(&ins[0].shape()[*axis + 1..]);
            let view = ins[0].to_f32()?;
            let out = super::gather::gather(
                |i| view[i],
                ins[0].shape(),
                |p| index_values[p],
                index_values.len(),
                ins[1].shape(),
                *axis,
                NO_EQN,
            )?;
            Ok(HostTensor::f32_arc(out_shape, out))
        }
        OpKind::Scatter { .. } => scatter_axis0(&ins[0], &ins[1], NO_EQN),
        OpKind::Broadcast { shape } => broadcast_to(&ins[0], shape),
        other => Err(EvalError::unsupported(
            "apply_movement",
            format!("{} is not a movement op", other.name()),
        )),
    }
}

/// The storage-aware walk's movement over every carrier: host tensors through the functions above
/// (every dtype, in its own storage class), and a refusal for exact and packed carriers.
impl Value {
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, EvalError> {
        match self {
            Value::Host(t) => Ok(reshape(t, &shape)?.into()),
            Value::Owner(value) => value.reshape(shape).map(Value::Owner),
            Value::Packed(_) => Err(EvalError::Packed(PackedEvalError::WrongConsumer {
                operation: "Reshape",
            })),
        }
    }

    pub fn transpose(&self, perm: &[usize]) -> Result<Self, EvalError> {
        match self {
            Value::Host(t) => Ok(transpose(t, perm)?.into()),
            Value::Owner(value) => value.transpose(perm),
            Value::Packed(_) => Err(EvalError::Packed(PackedEvalError::WrongConsumer {
                operation: "Transpose",
            })),
        }
    }

    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Result<Self, EvalError> {
        match self {
            Value::Host(t) => Ok(slice(t, axis, start, end)?.into()),
            Value::Owner(value) => Err(value.unsupported("Slice")),
            Value::Packed(_) => Err(EvalError::Packed(PackedEvalError::WrongConsumer {
                operation: "Slice",
            })),
        }
    }

    pub fn concat(parts: &[&Self], axis: usize) -> Result<Self, EvalError> {
        let tensors = parts
            .iter()
            .map(|part| match part {
                Value::Host(tensor) => Ok(tensor),
                Value::Owner(value) => Err(value.unsupported("Concat")),
                Value::Packed(_) => Err(EvalError::Packed(PackedEvalError::WrongConsumer {
                    operation: "Concat",
                })),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(concat(&tensors, axis)?.into())
    }

    pub fn dynamic_update_slice(
        &self,
        update: &Self,
        index: usize,
        axis: usize,
    ) -> Result<Self, EvalError> {
        match (self, update) {
            (Value::Owner(value), _) | (_, Value::Owner(value)) => {
                Err(value.unsupported("DynamicUpdateSlice"))
            }
            (Value::Host(operand), Value::Host(update)) => {
                Ok(dynamic_update_slice(operand, update, index, axis)?.into())
            }
            _ => Err(EvalError::Packed(PackedEvalError::WrongConsumer {
                operation: "DynamicUpdateSlice",
            })),
        }
    }

    pub fn broadcast(&self, shape: Vec<usize>) -> Result<Self, EvalError> {
        match self {
            Value::Host(t) => Ok(broadcast_to(t, &shape)?.into()),
            Value::Owner(value) => Err(value.unsupported("Broadcast")),
            Value::Packed(_) => Err(EvalError::Packed(PackedEvalError::WrongConsumer {
                operation: "Broadcast",
            })),
        }
    }

    pub fn scatter_update(
        &self,
        src: &Self,
        inverse: &HostTensor,
        eqn: ValueId,
    ) -> Result<Self, EvalError> {
        match (self, src) {
            (Value::Owner(value), _) | (_, Value::Owner(value)) => {
                Err(value.unsupported("ScatterUpdate"))
            }
            (Value::Host(base), Value::Host(src)) => {
                Ok(scatter_update(base, src, inverse, base.shape(), eqn)?.into())
            }
            _ => Err(EvalError::Packed(PackedEvalError::WrongConsumer {
                operation: "ScatterUpdate",
            })),
        }
    }
}
