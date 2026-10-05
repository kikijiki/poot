//! Encoder and vision-language request processing.

pub mod encoder;
pub(crate) mod mrope;
pub(crate) mod qwen_vl_chat;
pub(crate) mod qwen_vl_image;
pub(crate) mod qwen_vl_mrope;
pub mod vlm;

/// The vision/text encoder weights of the multimodal paths: always dense tensors, read on the host,
/// and bound to a graph as `poot_eval::Value::Host` inputs. The Runner's own weights are a
/// constant-name map of `poot_eval::Value`s.
pub type DenseWeights = std::collections::HashMap<String, poot_tensor::HostTensor>;

/// A checkpoint tensor as the f32 tensor the encoder and VLM graphs (traced in f32) bind. A bf16/f16
/// weight is widened here, the one explicit decode on these paths; an f32 weight is returned as is.
pub(crate) fn materialize_f32(
    store: &poot_quant::weights::WeightStore,
    name: &str,
) -> crate::error::Result<poot_tensor::HostTensor> {
    let tensor = poot_eval::materialize_dense(store, name).map_err(|error| err!("{error}"))?;
    if tensor.dtype() == poot_tensor::DType::F32 {
        return Ok(tensor);
    }
    let values = tensor
        .to_f32()
        .map_err(|error| err!("{name}: {error}"))?
        .into_owned();
    Ok(poot_tensor::HostTensor::f32(
        tensor.shape().to_vec(),
        values,
    ))
}
