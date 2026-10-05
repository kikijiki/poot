//! A mapped weight as a component reads it, and the dense linear projection over one.
//!
//! A component never sees bytes or a storage choice: it holds [`Weight`]s (a typed id and its
//! handle from the model's [`WeightMap`]) and declares each as a graph const named
//! [`WeightId::const_name`], in checkpoint orientation (`[out, in]` for a projection). [`linear`]
//! spells every projection as the logical dense `matmul(x, transpose(w))`, whatever the handle's
//! format: `compile` claims that pattern into a dense contraction, and a packed handle's const is
//! rewritten into its `PackedDequant` chain by the one packed-weight transform
//! (`poot_graph_plan::bind_packed_weights` over `WeightFormats::from_weight_map`). There is no packed
//! arm here.

use poot_graph_ir::{BinOp, Builder, TensorType, Traced};
use poot_quant::weights::{HandleFormat, WeightHandle, WeightId, WeightMap, WeightMapError};
use poot_tensor::DType;

use crate::model::{FamilyKey, ModelError};

/// One weight a component reads: its typed id and handle, checked against the shape the component
/// needs when it is made, so tracing with it cannot fail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Weight {
    id: WeightId,
    handle: WeightHandle,
}

/// Why a mapped weight cannot serve the component that asked for it.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WeightError {
    #[error("weight {id} is not in the weight map")]
    Unmapped { id: WeightId },
    #[error("weight {id} has shape {found:?}, expected {expected:?}")]
    Shape {
        id: WeightId,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    #[error("weight {id} is stored as {format:?}, which is not a float weight")]
    Format { id: WeightId, format: HandleFormat },
}

impl WeightError {
    /// The model error `family` reports for this weight.
    pub fn for_family(self, family: FamilyKey) -> ModelError {
        match self {
            WeightError::Unmapped { id } => ModelError::Weight {
                family,
                source: WeightMapError::Unmapped { id },
            },
            WeightError::Shape {
                id,
                expected,
                found,
            } => ModelError::WeightShape {
                family,
                id,
                expected,
                found,
            },
            WeightError::Format { id, format } => ModelError::WeightFormat { family, id, format },
        }
    }
}

impl Weight {
    /// `id`'s handle from `map`, which must have `shape` and be a float weight: a dense F32, BF16 or
    /// F16 entry, or a packed one (whose logical decode is F32).
    pub fn new(map: &WeightMap, id: WeightId, shape: &[usize]) -> Result<Self, WeightError> {
        let handle = map.handle(id).ok_or(WeightError::Unmapped { id })?;
        if handle.shape != shape {
            return Err(WeightError::Shape {
                id,
                expected: shape.to_vec(),
                found: handle.shape.clone(),
            });
        }
        match handle.format {
            HandleFormat::Dense(DType::F32 | DType::BF16 | DType::F16)
            | HandleFormat::Packed(_) => {}
            format => return Err(WeightError::Format { id, format }),
        }
        Ok(Self {
            id,
            handle: handle.clone(),
        })
    }

    pub fn id(&self) -> WeightId {
        self.id
    }

    pub fn shape(&self) -> &[usize] {
        &self.handle.shape
    }

    /// Declare the weight as a graph const at its logical dtype: a dense entry's stored dtype, F32
    /// for a packed one (the packed-weight transform replaces the const with its decode).
    pub fn declare(&self, b: &Builder) -> Traced {
        let dtype = match self.handle.format {
            HandleFormat::Dense(dtype) => dtype,
            HandleFormat::Packed(_) => DType::F32,
        };
        b.constant(
            &self.id.const_name(),
            TensorType::new(self.handle.shape.clone(), dtype),
        )
    }

    /// Declare the weight read as F32 by elementwise arithmetic (a norm scale, a bias).
    pub fn declare_f32(&self, b: &Builder) -> Traced {
        let w = self.declare(b);
        if b.aval(w).dtype == DType::F32 {
            w
        } else {
            b.cast(w, DType::F32)
        }
    }
}

/// `x @ w^T (+ bias)` for an `[out, in]` weight: `x` is `[.., in]`, the result `[.., out]`.
pub fn linear(b: &Builder, x: Traced, w: &Weight, bias: Option<&Weight>) -> Traced {
    let wt = b.transpose(w.declare(b), vec![1, 0]);
    let y = b.matmul(x, wt);
    match bias {
        Some(bias) => {
            let shape = b.aval(y).shape;
            let bias = b.broadcast(bias.declare_f32(b), shape);
            b.binary(BinOp::Add, y, bias)
        }
        None => y,
    }
}

#[cfg(test)]
mod tests {
    use poot_quant::weights::WeightRole;

    use super::*;
    use crate::components::standard::oracle::{self, Fx};

    /// SC-001: `linear` against a hand-written f64 `x W^T + b`, with BF16 storage for the weight
    /// (the stored dtype is declared, not widened) and F32 for the bias.
    #[test]
    fn linear_matches_an_f64_reference_and_depends_on_weight_and_bias() {
        let (m, k, n) = (3, 5, 4);
        let w_id = WeightId::layer(0, WeightRole::Attn(poot_quant::weights::AttnRole::Q));
        let b_id = WeightId::layer(0, WeightRole::Attn(poot_quant::weights::AttnRole::QBias));
        let mut fx = Fx::new(7);
        fx.bf16(w_id, &[n, k]);
        fx.f32(b_id, &[n]);
        let x = oracle::random(m * k, 11);
        let run = |fx: &Fx| {
            let map = fx.map();
            let w = Weight::new(&map, w_id, &[n, k]).unwrap();
            let bias = Weight::new(&map, b_id, &[n]).unwrap();
            let b = Builder::new();
            let xs = b.constant("x", TensorType::f32(vec![m, k]));
            let y = linear(&b, xs, &w, Some(&bias));
            let g = b.finish(y);
            fx.eval(&g, &[("x", &x)], &[], &[]).0
        };
        let got = run(&fx);
        let (w, bias) = (fx.values(w_id), fx.values(b_id));
        let mut want = vec![0.0f64; m * n];
        for i in 0..m {
            for o in 0..n {
                let dot: f64 = (0..k)
                    .map(|j| f64::from(x[i * k + j]) * f64::from(w[o * k + j]))
                    .sum();
                want[i * n + o] = dot + f64::from(bias[o]);
            }
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        for id in [w_id, b_id] {
            let mut perturbed = fx.clone();
            perturbed.perturb(id);
            oracle::assert_differs(&run(&perturbed), &got, id);
        }
    }

    /// SC-004's component half: `linear` over a Q8_0 handle emits the same dense
    /// `matmul(x, transpose(w))` (an F32 const at the logical shape); the packed-weight transform,
    /// fed the map's formats, turns the const into its decode, and the oracle result is the f64
    /// product with the decoded weight.
    #[test]
    fn linear_over_a_packed_handle_is_the_dense_spelling_bound_by_the_transform() {
        use poot_graph_ir::OpKind;
        use poot_graph_plan::{WeightFormats, bind_packed_weights};
        use poot_quant::format::WeightFormat;

        let (m, k, n) = (2, 64, 8);
        let id = WeightId::layer(0, WeightRole::Ffn(poot_quant::weights::FfnRole::Up));
        let mut fx = Fx::new(1);
        fx.packed(
            id,
            poot_test_util::packed::random_payload(WeightFormat::Q8_0, [n, k], 17),
        );
        let map = fx.map();
        let w = Weight::new(&map, id, &[n, k]).unwrap();
        let b = Builder::new();
        let xs = b.constant("x", TensorType::f32(vec![m, k]));
        let y = linear(&b, xs, &w, None);
        let g = b.finish(y);
        let declared = g.aval(g.inputs[1]);
        assert_eq!(
            (declared.dtype, declared.shape.clone()),
            (DType::F32, vec![n, k])
        );
        let bound = bind_packed_weights(&g, &WeightFormats::from_weight_map(&map)).unwrap();
        assert!(
            bound
                .eqns
                .iter()
                .any(|e| matches!(e.op, OpKind::PackedDequant { .. }))
        );
        let x = oracle::random(m * k, 2);
        let got = fx.eval(&bound, &[("x", &x)], &[], &[]).0;
        let wv = fx.values(id);
        let want: Vec<f64> = (0..m)
            .flat_map(|i| {
                let (x, wv) = (&x, &wv);
                (0..n).map(move |o| {
                    (0..k)
                        .map(|j| f64::from(x[i * k + j]) * f64::from(wv[o * k + j]))
                        .sum::<f64>()
                })
            })
            .collect();
        oracle::assert_matches_f64(&got, &want, 1e-5);
    }

    #[test]
    fn a_weight_of_the_wrong_shape_or_dtype_is_refused_by_id() {
        let id = WeightId::model(WeightRole::FinalNorm);
        let mut fx = Fx::new(1);
        fx.f32(id, &[8]);
        let map = fx.map();
        assert_eq!(
            Weight::new(&map, id, &[4]),
            Err(WeightError::Shape {
                id,
                expected: vec![4],
                found: vec![8],
            })
        );
        let missing = WeightId::model(WeightRole::Head);
        assert_eq!(
            Weight::new(&map, missing, &[4]),
            Err(WeightError::Unmapped { id: missing })
        );
    }
}
