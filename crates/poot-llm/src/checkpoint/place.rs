//! Card 545a: how a loader places one stored weight under the constant name a tracer
//! declares.
//!
//! A tracer declares every weight dense (`[K, out]` for a projection, `[rows, K]` for an embedding
//! table, `[E, K, out]` for an expert stack). A checkpoint stores each weight as `[out, K]` rows,
//! dense or packed: a GGUF tensor, or a safetensors linear `pack_quantized_linears` packed from its
//! checkpoint bytes (Card 654: DeepSeek-3.2's block FP8). [`StoredRows`] is one stored `[out, K]`
//! weight of either kind, and every loader row op (a fused-projection slice, llama's q/k un-permute,
//! gate||up concatenation, gpt-oss's interleave) is one [`StoredRows::gather`] over whole stored
//! rows, or one [`StoredRows::concat`] of whole weights: packed rows move byte for byte
//! ([`PackedPayload::gather_rows`], [`PackedPayload::concat_rows`]), dense rows by value. Placing a
//! stored owner shares it (an `Arc`, no byte copy). [`WeightPlacement`] then records a dense
//! weight as the tracer's constant and a packed one as its source carriers plus the one
//! [`WeightFormats`] row that `bind_packed_weights` reads, so the formats are derived from the stored
//! payloads and never restated per family.

use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{Value, materialize_dense};
use poot_graph_ir::packed_source::PackedSourceName;
use poot_graph_plan::{PackedConst, PackedLayout, WeightFormats};
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_quant::{PackedComponentRef, PackedPayload};
use poot_tensor::HostTensor;

use super::gguf::permute::{gather, gather_from};
use super::gguf::transpose2d;
use crate::error::{OptionExt, Result, ResultExt};

/// One stored `[out, K]` weight, dense or packed.
#[derive(Clone, Debug)]
pub(crate) enum StoredRows {
    Dense(HostTensor),
    Packed(Arc<PackedPayload>),
}

impl StoredRows {
    /// The stored weight `name`: its packed payload shared (no byte copy), or its dense values.
    pub(crate) fn read(store: &WeightStore, name: &str) -> Result<Self> {
        match store.get(name) {
            Some(WeightEntry::Packed(payload)) => Ok(Self::Packed(Arc::clone(payload))),
            _ => materialize_dense(store, name)
                .map(Self::Dense)
                .with_context(|| format!("materialize {name}")),
        }
    }

    /// The experts of a GGUF expert tensor `name`: `read_gguf` keeps a packed `[E, out, K]` tensor
    /// as one owner per expert (`"{name}.{e}"`), a dense one whole (split here by value).
    pub(crate) fn experts(store: &WeightStore, name: &str) -> Result<Vec<Self>> {
        if !store.contains(name) {
            let owners: Vec<Self> = (0..)
                .map(|e| format!("{name}.{e}"))
                .take_while(|owner| store.contains(owner))
                .map(|owner| Self::read(store, &owner))
                .collect::<Result<_>>()?;
            if owners.is_empty() {
                bail!("gguf expert tensor {name} is missing");
            }
            return Ok(owners);
        }
        let stacked =
            materialize_dense(store, name).with_context(|| format!("materialize {name}"))?;
        let [e, out, k] = stacked.shape()[..] else {
            bail!(
                "gguf expert tensor {name}: shape {:?} is not [E, out, K]",
                stacked.shape()
            );
        };
        Ok((0..e)
            .map(|x| {
                Self::Dense(gather(
                    &stacked,
                    vec![out, k],
                    x * out * k..(x + 1) * out * k,
                ))
            })
            .collect())
    }

    pub(crate) fn rows(&self) -> usize {
        match self {
            Self::Dense(tensor) => tensor.shape()[0],
            Self::Packed(payload) => payload.weight().shape()[0],
        }
    }

    /// Rows `rows` of `parts` stacked in order (row `r` of the stack is row `r - offset` of the
    /// part holding it). Every part must be the same kind and width.
    pub(crate) fn gather(parts: &[&Self], rows: &[usize]) -> Result<Self> {
        if let Some(packed) = parts
            .iter()
            .map(|part| match part {
                Self::Packed(payload) => Some(payload.as_ref()),
                Self::Dense(_) => None,
            })
            .collect::<Option<Vec<&PackedPayload>>>()
        {
            return PackedPayload::gather_rows(&packed, rows)
                .map(|payload| Self::Packed(Arc::new(payload)))
                .context("gather packed rows");
        }
        let dense: Vec<&HostTensor> = parts
            .iter()
            .map(|part| match part {
                Self::Dense(tensor) => Ok(tensor),
                Self::Packed(payload) => Err(err!(
                    "cannot gather rows across a dense and a packed ({:?}) weight",
                    payload.weight().format()
                )),
            })
            .collect::<Result<_>>()?;
        let k = dense.first().map_or(0, |tensor| tensor.shape()[1]);
        if let Some(other) = dense.iter().find(|tensor| tensor.shape()[1] != k) {
            bail!("cannot gather rows of width {k} and {}", other.shape()[1]);
        }
        let dtype = dense.first().map(|tensor| tensor.dtype());
        if let Some(other) = dense.iter().find(|tensor| Some(tensor.dtype()) != dtype) {
            bail!(
                "cannot gather rows of {} and {} weights",
                dtype.map_or_else(String::new, |dtype| dtype.to_string()),
                other.dtype()
            );
        }
        let stacked: usize = dense.iter().map(|tensor| tensor.shape()[0]).sum();
        // Each destination row is one stacked source row: (part holding it, offset within the part).
        let mut sources = Vec::with_capacity(rows.len());
        for &row in rows {
            let mut local = row;
            let part = dense
                .iter()
                .position(|tensor| {
                    if local < tensor.shape()[0] {
                        return true;
                    }
                    local -= tensor.shape()[0];
                    false
                })
                .with_context(|| format!("row {row} is past the {stacked} stacked rows"))?;
            sources.push((part, local));
        }
        Ok(Self::Dense(gather_from(
            &dense,
            vec![rows.len(), k],
            sources
                .into_iter()
                .flat_map(|(part, local)| (0..k).map(move |col| (part, local * k + col))),
        )))
    }

    /// Rows `lo..hi`.
    pub(crate) fn slice(&self, lo: usize, hi: usize) -> Result<Self> {
        Self::gather(&[self], &(lo..hi).collect::<Vec<_>>())
    }

    /// `parts` stacked whole, in order (a fused `gate||up` from `gate_proj` and `up_proj`). Packed
    /// parts go through [`PackedPayload::concat_rows`], which also stacks a planar safetensors
    /// format (block FP8) whose blocks stay whole, and refuses one whose blocks would straddle the
    /// boundary; dense parts stack by value.
    pub(crate) fn concat(parts: &[&Self]) -> Result<Self> {
        if let Some(packed) = parts
            .iter()
            .map(|part| match part {
                Self::Packed(payload) => Some(payload.as_ref()),
                Self::Dense(_) => None,
            })
            .collect::<Option<Vec<&PackedPayload>>>()
        {
            return PackedPayload::concat_rows(&packed)
                .map(|payload| Self::Packed(Arc::new(payload)))
                .context("concatenate packed rows");
        }
        let rows: usize = parts.iter().map(|part| part.rows()).sum();
        Self::gather(parts, &(0..rows).collect::<Vec<_>>())
    }
}

/// The row order that undoes llama.cpp's q/k rope permute. `convert_hf_to_gguf.py`'s
/// `LlamaModel.permute` stores each head's rows as `reshape(heads, 2, half, K).swapaxes(1, 2)`: stored
/// row `h*hd + 2*j + t` is HF row `h*hd + t*half + j`. So HF row `h*hd + j` is stored row
/// `h*hd + 2*j`, and HF row `h*hd + half + j` is stored row `h*hd + 2*j + 1`.
pub(crate) fn unpermute_qk_rows(out: usize, heads: usize) -> Vec<usize> {
    let hd = out / heads;
    let half = hd / 2;
    let mut rows = vec![0; out];
    for h in 0..heads {
        let base = h * hd;
        for j in 0..half {
            rows[base + j] = base + 2 * j;
            rows[base + half + j] = base + 2 * j + 1;
        }
    }
    rows
}

/// `[0, n, 1, n+1, ...]`: two `n`-row parts interleaved row by row (gpt-oss's gate/up).
pub(crate) fn interleave_rows(n: usize) -> Vec<usize> {
    (0..n).flat_map(|i| [i, n + i]).collect()
}

/// A model's weights as the tracers name them, with the packed-storage record
/// `bind_packed_weights` reads.
#[derive(Default)]
pub(crate) struct WeightPlacement {
    weights: HashMap<String, Value>,
    formats: WeightFormats,
}

impl WeightPlacement {
    /// A dense constant as it is (norms, biases, rope tables).
    pub(crate) fn dense(&mut self, name: impl Into<String>, tensor: HostTensor) {
        self.weights.insert(name.into(), Value::from(tensor));
    }

    /// A projection the tracer reads as `[K, out]` from a stored `[out, K]` weight.
    pub(crate) fn projection(&mut self, name: impl Into<String>, rows: StoredRows) -> Result<()> {
        let name = name.into();
        match rows {
            StoredRows::Dense(tensor) => {
                self.dense(name, transpose2d(&tensor));
                Ok(())
            }
            StoredRows::Packed(payload) => self.packed(name, vec![payload], PackedLayout::Columns),
        }
    }

    /// A table the tracer gathers rows of, `[rows, K]` as stored.
    pub(crate) fn table(&mut self, name: impl Into<String>, rows: StoredRows) -> Result<()> {
        let name = name.into();
        match rows {
            StoredRows::Dense(tensor) => {
                self.dense(name, tensor);
                Ok(())
            }
            StoredRows::Packed(payload) => self.packed(name, vec![payload], PackedLayout::Rows),
        }
    }

    /// An expert stack the tracer reads as `[E, K, out]`, one stored `[out, K]` weight per expert.
    pub(crate) fn experts(
        &mut self,
        name: impl Into<String>,
        experts: Vec<StoredRows>,
    ) -> Result<()> {
        let name = name.into();
        let packed: Option<Vec<Arc<PackedPayload>>> = experts
            .iter()
            .map(|expert| match expert {
                StoredRows::Packed(payload) => Some(Arc::clone(payload)),
                StoredRows::Dense(_) => None,
            })
            .collect();
        if let Some(packed) = packed {
            return self.packed(name, packed, PackedLayout::StackedColumns);
        }
        let dense: Vec<HostTensor> = experts
            .into_iter()
            .map(|expert| match expert {
                StoredRows::Dense(tensor) => Ok(transpose2d(&tensor)),
                StoredRows::Packed(payload) => Err(err!(
                    "{name}: expert stack mixes dense and packed ({:?}) experts",
                    payload.weight().format()
                )),
            })
            .collect::<Result<_>>()?;
        let slice = dense
            .first()
            .map(|t| t.shape().to_vec())
            .unwrap_or_default();
        let mut shape = vec![dense.len()];
        shape.extend_from_slice(&slice);
        let sources: Vec<&HostTensor> = dense.iter().collect();
        let per_expert: usize = slice.iter().product();
        let dtype = dense.first().map(HostTensor::dtype);
        if let Some(other) = dense.iter().find(|t| Some(t.dtype()) != dtype) {
            bail!(
                "{name}: expert stack mixes {dtype:?} and {} experts",
                other.dtype()
            );
        }
        self.dense(
            name,
            gather_from(
                &sources,
                shape,
                (0..dense.len()).flat_map(|e| (0..per_expert).map(move |k| (e, k))),
            ),
        );
        Ok(())
    }

    fn packed(
        &mut self,
        name: String,
        owners: Vec<Arc<PackedPayload>>,
        layout: PackedLayout,
    ) -> Result<()> {
        place_packed(&mut self.weights, &mut self.formats, name, owners, layout)
    }

    pub(crate) fn finish(self) -> (HashMap<String, Value>, WeightFormats) {
        (self.weights, self.formats)
    }
}

/// Record constant `name` as stored packed in `owners` (one, or one per expert): each owner's source
/// carriers go into `weights` under their `PackedSourceName`, and one `formats` row names them.
pub(crate) fn place_packed(
    weights: &mut HashMap<String, Value>,
    formats: &mut WeightFormats,
    name: String,
    owners: Vec<Arc<PackedPayload>>,
    layout: PackedLayout,
) -> Result<()> {
    let base = name.strip_suffix(".weight").unwrap_or(&name).to_string();
    let Some(weight) = owners.first().map(|owner| owner.weight()) else {
        bail!("{name}: a packed weight needs at least one owner");
    };
    if let Some(other) = owners.iter().find(|owner| owner.weight() != weight) {
        bail!(
            "{name}: expert owners disagree: {weight:?} vs {:?}",
            other.weight()
        );
    }
    let linear_ids: Vec<String> = match layout {
        PackedLayout::StackedColumns => (0..owners.len()).map(|e| format!("{base}.{e}")).collect(),
        PackedLayout::Rows | PackedLayout::Columns => vec![base],
    };
    for (linear_id, owner) in linear_ids.iter().zip(&owners) {
        for role in weight.sources() {
            weights.insert(
                PackedSourceName::new(linear_id, role).into(),
                Value::Packed(PackedComponentRef::new(Arc::clone(owner), role)),
            );
        }
    }
    formats
        .insert(
            name,
            PackedConst {
                linear_ids,
                weight,
                layout,
            },
        )
        .context("record packed weight")
}
