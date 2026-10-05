//! Vision-language model loading (spec 049 slice 4, card 054): reads a SmolVLM / idefics3 checkpoint's three
//! sub-models (SigLIP vision encoder, connector projection, llama-class text model) from safetensors
//! into the named `Tensor` map the vision-encoder graph (`poot_models::vision`) binds.

use std::collections::HashMap;

use poot_eval::Value;
use poot_load::safetensors;
use poot_models::vision::VisionConfig;
use poot_quant::weights::WeightStore;
use poot_tensor::HostTensor;

use super::{DenseWeights, materialize_f32};
use crate::GenerationControl;
use crate::checkpoint::gguf::rope::rope_tables;
use crate::checkpoint::gguf::transpose2d;
use crate::core::generate::{MAX_PREFILL_CAP, argmax};
use crate::core::graphs::causal_mask;
use crate::error::Result;
use crate::text::tokenize::stream_piece_delta;

/// The greedy next token from one row of logits. A non-finite row is a [`crate::SamplerFault`], not a token
/// and not a panic (the old local `partial_cmp().unwrap()` maximum panicked on NaN).
fn greedy_token(logits: &[f32]) -> Result<u32> {
    Ok(argmax(logits)? as u32)
}

/// Loads the SigLIP vision encoder weights from a SmolVLM checkpoint directory (spec 049 slice 4b).
/// Returns the `name -> Tensor` map: `patch_w` (the conv2d `[H,3,P,P]` weight reshaped and transposed to
/// the linear `[C*P*P, H]` layout), `patch_b`, `pos_embed`, per-layer block weights (`l1w{i}`/`wq{i}`/...
/// as in the tests), and `post_ln_w`/`post_ln_b`. bf16 weights are already f32 (the loader converts on
/// read).
pub fn load_vision_weights(
    dir: impl AsRef<std::path::Path>,
    cfg: &VisionConfig,
) -> Result<DenseWeights> {
    let st = safetensors::load_weight_store(dir).map_err(|e| err!("load safetensors: {e}"))?;
    let mut m = DenseWeights::new();
    let get = |st: &WeightStore, name: &str| -> Result<HostTensor> {
        materialize_f32(st, name).map_err(|e| err!("missing vision weight {name}: {e}"))
    };
    let v = "model.vision_model";

    // patch embedding: conv weight [H, C, P, P] = [out, (c,i,j)=in] -> transpose to the linear [in, out].
    let pw = materialize_f32(&st, &format!("{v}.embeddings.patch_embedding.weight"))
        .map_err(|e| err!("patch weight: {e}"))?;
    let out = cfg.hidden();
    let inp = cfg.channels * cfg.patch_size * cfg.patch_size;
    let patch_rows = pw
        .reshaped(vec![out, inp])
        .map_err(|e| err!("patch weight: {e}"))?;
    m.insert("patch_w".into(), transpose2d(&patch_rows));
    m.insert(
        "patch_b".into(),
        get(&st, &format!("{v}.embeddings.patch_embedding.bias"))?,
    );
    m.insert(
        "pos_embed".into(),
        get(&st, &format!("{v}.embeddings.position_embedding.weight"))?,
    );

    // the linear weights are HF `[out, in]`; poot's matmul wants `[in, out]`, so transpose each.
    let lin = |st: &WeightStore, name: &str| -> Result<HostTensor> {
        let t = materialize_f32(st, name).map_err(|e| err!("missing {name}: {e}"))?;
        Ok(transpose2d(&t))
    };

    for l in 0..cfg.layers {
        let p = format!("{v}.encoder.layers.{l}");
        m.insert(
            format!("l1w{l}"),
            get(&st, &format!("{p}.layer_norm1.weight"))?,
        );
        m.insert(
            format!("l1b{l}"),
            get(&st, &format!("{p}.layer_norm1.bias"))?,
        );
        m.insert(
            format!("wq{l}"),
            lin(&st, &format!("{p}.self_attn.q_proj.weight"))?,
        );
        m.insert(
            format!("bq{l}"),
            get(&st, &format!("{p}.self_attn.q_proj.bias"))?,
        );
        m.insert(
            format!("wk{l}"),
            lin(&st, &format!("{p}.self_attn.k_proj.weight"))?,
        );
        m.insert(
            format!("bk{l}"),
            get(&st, &format!("{p}.self_attn.k_proj.bias"))?,
        );
        m.insert(
            format!("wv{l}"),
            lin(&st, &format!("{p}.self_attn.v_proj.weight"))?,
        );
        m.insert(
            format!("bv{l}"),
            get(&st, &format!("{p}.self_attn.v_proj.bias"))?,
        );
        m.insert(
            format!("wo{l}"),
            lin(&st, &format!("{p}.self_attn.out_proj.weight"))?,
        );
        m.insert(
            format!("bo{l}"),
            get(&st, &format!("{p}.self_attn.out_proj.bias"))?,
        );
        m.insert(
            format!("l2w{l}"),
            get(&st, &format!("{p}.layer_norm2.weight"))?,
        );
        m.insert(
            format!("l2b{l}"),
            get(&st, &format!("{p}.layer_norm2.bias"))?,
        );
        m.insert(format!("f1w{l}"), lin(&st, &format!("{p}.mlp.fc1.weight"))?);
        m.insert(format!("f1b{l}"), get(&st, &format!("{p}.mlp.fc1.bias"))?);
        m.insert(format!("f2w{l}"), lin(&st, &format!("{p}.mlp.fc2.weight"))?);
        m.insert(format!("f2b{l}"), get(&st, &format!("{p}.mlp.fc2.bias"))?);
    }
    m.insert(
        "plw".into(),
        get(&st, &format!("{v}.post_layernorm.weight"))?,
    );
    m.insert("plb".into(), get(&st, &format!("{v}.post_layernorm.bias"))?);
    Ok(m)
}

/// Loads the connector's modality-projection weight (spec 049 slice 4b): HF `proj.weight [text_hidden,
/// vision_hidden*scale^2]` transposed to poot's linear `[in, out]` layout `[vision_hidden*scale^2,
/// text_hidden]`.
pub fn load_connector_weight(dir: impl AsRef<std::path::Path>) -> Result<HostTensor> {
    let st = safetensors::load_weight_store(dir).map_err(|e| err!("load safetensors: {e}"))?;
    let t = materialize_f32(&st, "model.connector.modality_projection.proj.weight")
        .map_err(|e| err!("connector weight: {e}"))?;
    Ok(transpose2d(&t)) // [text_hidden, in] -> [in, text_hidden]
}

/// Shape params for the llama-class text decoder embedded in a VLM (from the checkpoint's `text_config`).
#[derive(Clone, Copy, Debug)]
pub struct TextConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub q_dim: usize,  // n_heads * head_dim
    pub kv_dim: usize, // n_kv_heads * head_dim
    pub intermediate: usize,
}

impl TextConfig {
    /// SmolVLM-256m's text model (SmolLM2-class llama): vocab 49280, hidden 576, 30 layers, 9 heads x 64 (q),
    /// 3 KV heads x 64 (kv 192), MLP 1536.
    pub fn smolvlm() -> Self {
        TextConfig {
            vocab: 49280,
            hidden: 576,
            layers: 30,
            q_dim: 576,
            kv_dim: 192,
            intermediate: 1536,
        }
    }
}

/// Upper bound on decoded pixel count, guarding against a decompression bomb (a tiny file that decodes
/// to huge dimensions, blowing up the `w*h*3` RGB buffer and its 4x f32 expansion). ~32 MP (e.g.
/// 8192x4096) is far above any real SmolVLM input. The `image` crate's default `Limits` (~512MB soft
/// `max_alloc`) is not honored by every decoder, so the header dimensions are checked directly.
const MAX_IMAGE_PIXELS: u64 = 32 * 1024 * 1024;

/// Upper bound on the total idefics3 frames across all images in one caption request. Each image tiles
/// into ~17-34 frames (~3 MB of pixels each), all held in memory before the vision tower runs, and the
/// image count is client-controlled (the per-image pixel cap does not bound it). 512 frames (~15-30
/// images, ~1.5 GB peak) is far above any real multi-image chat.
const MAX_VLM_FRAMES: usize = 512;

fn check_image_dims(w: u32, h: u32) -> Result<()> {
    let px = (w as u64) * (h as u64);
    if px > MAX_IMAGE_PIXELS {
        return Err(err!(
            "image {w}x{h} ({px} pixels) exceeds the {MAX_IMAGE_PIXELS}-pixel cap"
        ));
    }
    Ok(())
}

/// Decodes PNG/JPEG bytes into interleaved RGB `[0,255]` f32 + `(height, width)`, the input
/// [`VlmRunner::caption`] / [`poot_models::vision::preprocess_image`] expect (card 054). Alpha is dropped.
pub fn decode_image(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize)> {
    use std::io::Cursor;
    // Cheap header-only dimension read; reject an oversized image before decoding its pixels.
    let (w0, h0) = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| err!("decode image: {e}"))?
        .into_dimensions()
        .map_err(|e| err!("decode image dims: {e}"))?;
    check_image_dims(w0, h0)?;
    let img = image::load_from_memory(bytes)
        .map_err(|e| err!("decode image: {e}"))?
        .to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgb: Vec<f32> = img.into_raw().into_iter().map(|b| b as f32).collect();
    Ok((rgb, h, w))
}

/// Decodes an OpenAI `image_url` value, either a `data:image/...;base64,<b64>` data URL or raw base64,
/// into RGB (card 054 serving). Remote `http(s)` URLs are the caller's concern (no network in the
/// engine crate).
pub fn decode_image_data_url(url: &str) -> Result<(Vec<f32>, usize, usize)> {
    use base64::Engine;
    let b64 = match url.split_once(";base64,") {
        Some((_, data)) => data,
        None => url, // allow a bare base64 string
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| err!("base64 decode image_url: {e}"))?;
    decode_image(&bytes)
}

/// A loaded vision-language model (SmolVLM / idefics3): the three sub-models (SigLIP vision, connector,
/// llama-class text decoder) + tokenizer + RoPE, ready to caption images (spec 049 slice 4c).
pub struct VlmRunner {
    vcfg: poot_models::vision::VisionConfig,
    cfg: poot_models::qwen2::Qwen2Config,
    vision_w: DenseWeights,
    connector_w: HostTensor,
    text_w: DenseWeights,
    cos: HostTensor,
    sin: HostTensor,
    tokenizer: tokenizers::Tokenizer,
    scale: usize,
    image_token: u32,
    eos: u32,
}

impl VlmRunner {
    /// Load a SmolVLM-256m / idefics3 checkpoint directory (all three sub-models + tokenizer + RoPE tables).
    pub fn load(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        use poot_models::qwen2::Qwen2Config;
        use poot_models::vision::VisionConfig;
        let dir = dir.as_ref();
        let vcfg = VisionConfig::smolvlm();
        let tcfg = TextConfig::smolvlm();
        let cfg = Qwen2Config {
            vocab: tcfg.vocab,
            hidden: tcfg.hidden,
            inter: tcfg.intermediate,
            layers: tcfg.layers,
            n_heads: 9,
            n_kv_heads: 3,
            head_dim: 64,
            rotary_dim: 64,
            eps: 1e-5,
            max_pos: 8192,
            qkv_bias: false,
            qk_norm: false,
            ..Default::default()
        };
        let (cos, sin) = rope_tables(&cfg, 100000.0, None, None);
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| err!("load tokenizer: {e}"))?;
        Ok(VlmRunner {
            vcfg,
            cfg,
            vision_w: load_vision_weights(dir, &vcfg)?,
            connector_w: load_connector_weight(dir)?,
            text_w: load_text_weights(dir, &tcfg)?,
            cos,
            sin,
            tokenizer,
            scale: 4,
            image_token: 49190,
            eos: 2, // <end_of_utterance>
        })
    }

    /// The number of visual tokens one image produces (the connector output count = `num_patches / scale^2`).
    pub fn n_image_tokens(&self) -> usize {
        self.vcfg.num_patches() / (self.scale * self.scale)
    }

    /// Bakes this VLM's three sub-models (vision tower, connector, text) into one
    /// [`poot_quant::weights::WeightStore`] for the executor contract, named exactly as
    /// [`Self::caption_streaming_multi`]'s vision/splice/prefill/decode graphs bind their consts
    /// (the connector projection as `"proj"`, the RoPE tables as `"rope.cos"`/`"rope.sin"`, every
    /// other weight under its own stored name). The returned `ExecutableId` is reusable across
    /// many `caption_streaming_multi` calls (the poot-serve VLM thread owns one), since
    /// `load_weights` records the store once and uploads nothing until an entry actually binds a
    /// const.
    pub fn load_on(
        &self,
        exec: &mut dyn poot_executor::Executor,
    ) -> Result<poot_executor::ExecutableId> {
        let mut weights: HashMap<String, poot_eval::Value> = HashMap::new();
        for (name, t) in &self.vision_w {
            weights.insert(name.clone(), Value::Host(t.clone()));
        }
        weights.insert("proj".into(), Value::Host(self.connector_w.clone()));
        for (name, t) in &self.text_w {
            weights.insert(name.clone(), Value::Host(t.clone()));
        }
        weights.insert("rope.cos".into(), Value::Host(self.cos.clone()));
        weights.insert("rope.sin".into(), Value::Host(self.sin.clone()));
        let store = crate::core::runner::weight_store(&weights)?;
        exec.load_weights(
            std::sync::Arc::new(store),
            poot_executor::WeightSource::ConstNames,
        )
        .map_err(|error| err!("vlm load_on: {error}"))
    }

    /// Captions an image (greedy) and returns the full caption. See [`Self::caption_streaming`] for the
    /// token-incremental variant; this has no per-token callback.
    #[allow(clippy::too_many_arguments)]
    pub fn caption(
        &self,
        rgb: &[f32],
        h: usize,
        w: usize,
        question: &str,
        max_new: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
    ) -> Result<String> {
        self.caption_streaming(rgb, h, w, question, max_new, exec, exe, |_| {
            std::ops::ControlFlow::Continue(())
        })
    }

    /// Captions an image: full VLM forward (vision tower -> connector -> splice into the prompt's text
    /// embeddings -> text prefill from the spliced embeddings -> greedy decode), calling `on_token` with each
    /// newly decoded text piece (for SSE streaming) and returning the full caption. `rgb` is interleaved RGB
    /// in `[0,255]`, `h`x`w`; `question` is the user turn. Greedy, up to `max_new` tokens, on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn caption_streaming(
        &self,
        rgb: &[f32],
        h: usize,
        w: usize,
        question: &str,
        max_new: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<String> {
        self.caption_streaming_multi(&[(rgb, h, w)], question, max_new, exec, exe, on_token)
    }

    /// Captions one or more images in a single chat turn (the OpenAI vision API allows several `image_url`
    /// parts per message). Each image is split into idefics3 frames independently; all frames run through
    /// the vision tower, their visual tokens are concatenated in image-then-frame order and spliced into the
    /// multi-image prompt (one `<image>` block per image). `images` is a slice of `(rgb, h, w)` (interleaved
    /// RGB in `[0,255]`); the rest mirrors [`Self::caption_streaming`], the single-image case. Greedy, up to
    /// `max_new` tokens, on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn caption_streaming_multi(
        &self,
        images: &[(&[f32], usize, usize)],
        question: &str,
        max_new: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        mut on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<String> {
        use crate::backends::gpu_generate::{executor_target, slot_step_inputs, staged_program};
        use poot_graph_ir::builder::Builder;
        use poot_graph_ir::types::TensorType;
        use poot_graph_ir::{Slot, Storage, ValueId};
        use poot_models::qwen2::{trace_decode_kv_masked, trace_prefill_kv_embeds};
        use poot_models::vision::{
            PreprocessConfig, declare_vision_constants, idefics3_preprocess,
            image_splice_inverse_map, trace_connector, trace_image_splice, trace_vision_encoder,
        };
        if images.is_empty() {
            bail!("caption requires at least one image");
        }
        let th = self.cfg.hidden;
        let n_vis = self.n_image_tokens();
        // idefics3 image splitting: each image -> its own N frames (aspect-preserving tiles + a global
        // thumbnail), each `[C, S, S]`. Collect all frames across all images (image-then-frame order) plus each
        // image's tile grid. The prompt has one <image> block per image, so the total <image> placeholder count
        // == (sum of per-image frame counts) * n_vis == the concatenated visual-token count.
        let mut all_frames: Vec<Vec<f32>> = Vec::new();
        let mut grids: Vec<(usize, usize)> = Vec::new();
        for &(rgb, h, w) in images {
            let f = idefics3_preprocess(rgb, h, w, &PreprocessConfig::smolvlm());
            grids.push((f.rows, f.cols));
            all_frames.extend(f.frames);
            // Cap the total frames (checked incrementally, so the huge Vec is never materialized): the per-image
            // frame count is bounded but the image count is client-controlled, and each image is ~17-34 frames
            // (~3 MB of pixels each) held in `all_frames` before the vision tower runs.
            if all_frames.len() > MAX_VLM_FRAMES {
                bail!(
                    "too many image frames ({}+); at most {MAX_VLM_FRAMES} (idefics3 tiles each image into ~17-34 frames)",
                    all_frames.len()
                );
            }
        }
        let n_frames = all_frames.len();
        let prompt = smolvlm_caption_prompt_multi(question, &grids, n_vis);
        let ids: Vec<u32> = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| err!("encode prompt: {e}"))?
            .get_ids()
            .to_vec();
        let n = ids.len();
        // bound the KV span so an oversized prompt is a clean error, not an OOM abort (see
        // MAX_PREFILL_CAP; VLM sizes the span as `n + max_new`).
        let cap = n
            .checked_add(max_new)
            .filter(|&c| c <= MAX_PREFILL_CAP)
            .ok_or_else(|| {
                err!(
                    "prompt ({n}) + generation ({max_new}) exceeds the {}-token limit",
                    MAX_PREFILL_CAP
                )
            })?;
        let inv = image_splice_inverse_map(&ids, self.image_token);
        debug_assert_eq!(
            inv.iter().filter(|&&x| x >= 0.0).count(),
            n_frames * n_vis,
            "one <image> placeholder per visual token across all frames of all images"
        );

        // 1) vision tower: run each frame through the SigLIP encoder + connector to get n_vis visual tokens.
        // The single-frame vision graph is built once (one contract entry) and stepped per frame, which
        // keeps compile time and device memory bounded vs tracing all N encoders into one graph. Per-frame
        // visual tokens are concatenated on host in frame order (tiles row-major then global), matching
        // the prompt's <image> block order. `pixels` is the one per-step input (a `Slot`, not a `Const`:
        // every other VLM caption on this executable's long-lived entry sees a different frame); every
        // other vision/connector weight resolves from `Self::load_on`'s baked WeightStore by name.
        let vb = Builder::new();
        let pixels = vb.slot_named(
            Slot::Activation,
            "pixels",
            TensorType::f32(vec![
                self.vcfg.channels,
                self.vcfg.image_size,
                self.vcfg.image_size,
            ]),
        );
        let (vw, _binds) = declare_vision_constants(&vb, &self.vcfg);
        let conn_c = vb.constant(
            "proj",
            TensorType::f32(vec![self.vcfg.hidden() * self.scale * self.scale, th]),
        );
        let feats = trace_vision_encoder(&vb, pixels, &vw, &self.vcfg);
        let visual = trace_connector(&vb, feats, conn_c, self.vcfg.hidden(), self.scale);
        let visual = vb.reshape(visual, vec![n_vis, th]);
        let gvis = vb.finish(visual);
        let gvis_staged = staged_program(&gvis, executor_target(exec))
            .map_err(|e| err!("vlm vision graph staging: {e}"))?;
        let gvis_entry = exec
            .add_entry(exe, &gvis_staged)
            .map_err(|e| err!("vlm vision add_entry: {e}"))?;
        let mut visual_all = Vec::with_capacity(n_frames * n_vis * th);
        let vision_result: Result<()> = (|| {
            for frame in &all_frames {
                let mut bound: HashMap<ValueId, Value> = HashMap::new();
                bound.insert(
                    pixels.id,
                    HostTensor::f32(
                        vec![
                            self.vcfg.channels,
                            self.vcfg.image_size,
                            self.vcfg.image_size,
                        ],
                        frame.clone(),
                    )
                    .into(),
                );
                let inputs = slot_step_inputs(&gvis, &bound)?;
                let bytes = exec
                    .step(exe, gvis_entry, &inputs, &mut poot_executor::NoSync)
                    .map_err(|e| err!("vision forward: {e}"))?
                    .read()
                    .map_err(|e| err!("vision forward readback: {e}"))?;
                let v: &[f32] = bytemuck::cast_slice(&bytes);
                visual_all.extend_from_slice(v);
            }
            Ok(())
        })();
        match exec.remove_entry(exe, gvis_entry) {
            Ok(()) => vision_result?,
            Err(e) if vision_result.is_ok() => return Err(err!("vlm vision remove_entry: {e}")),
            Err(_) => vision_result?,
        }

        // 2) splice -> input embeddings [n, th]: gather the text embeddings for the prompt tokens (on-device,
        // dtype-safe) and scatter the concatenated visual tokens into the <image> placeholder positions.
        // One-shot entry, run exactly once per caption; `embed` is the only true constant here (named to
        // resolve from the same baked store entry the text decode graphs below bind under the same name).
        let sb = Builder::new();
        let embed_c = sb.constant(
            "model.embed_tokens.weight",
            TensorType::f32(vec![self.cfg.vocab, th]),
        );
        let tokens_c = sb.slot_named(Slot::Activation, "vlm.splice.tok", TensorType::f32(vec![n]));
        let inv_c = sb.slot_named(Slot::Activation, "vlm.splice.inv", TensorType::f32(vec![n]));
        let visual_c = sb.slot_named(
            Slot::Activation,
            "vlm.splice.visual",
            TensorType::f32(vec![n_frames * n_vis, th]),
        );
        let text_embeds = sb.gather(embed_c, 0, tokens_c);
        let spliced = trace_image_splice(&sb, text_embeds, visual_c, inv_c);
        let gsplice = sb.finish(spliced);
        let gsplice_staged = staged_program(&gsplice, executor_target(exec))
            .map_err(|e| err!("vlm splice graph staging: {e}"))?;
        let gsplice_entry = exec
            .add_entry(exe, &gsplice_staged)
            .map_err(|e| err!("vlm splice add_entry: {e}"))?;
        let splice_result: Result<Vec<u8>> = (|| {
            let mut bound: HashMap<ValueId, Value> = HashMap::new();
            bound.insert(
                tokens_c.id,
                HostTensor::f32(vec![n], ids.iter().map(|&t| t as f32).collect()).into(),
            );
            bound.insert(inv_c.id, HostTensor::f32(vec![n], inv).into());
            bound.insert(
                visual_c.id,
                HostTensor::f32(vec![n_frames * n_vis, th], visual_all).into(),
            );
            let inputs = slot_step_inputs(&gsplice, &bound)?;
            exec.step(exe, gsplice_entry, &inputs, &mut poot_executor::NoSync)
                .map_err(|e| err!("splice: {e}"))?
                .read()
                .map_err(|e| err!("splice readback: {e}"))
        })();
        let input_embeds_bytes = match exec.remove_entry(exe, gsplice_entry) {
            Ok(()) => splice_result?,
            Err(e) if splice_result.is_ok() => return Err(err!("vlm splice remove_entry: {e}")),
            Err(_) => splice_result?,
        };
        let input_embeds = HostTensor::f32(
            vec![n, th],
            bytemuck::cast_slice::<u8, f32>(&input_embeds_bytes).to_vec(),
        );

        // 2) prefill from the spliced embeds.
        //
        // `compile` (card 535b) with full fusion (flash-fuses prefill attention, so the `[1,Hq,n,n]`
        // softmax(QK^T) matrix is not materialized for the whole prompt; card 189, cf. card 183). Mirrors
        // `Runner::generate_kv_gpu_prefilled`'s prefill-then-decode handoff (card 546b): the prefill entry
        // is removed right after its one step (R-546-3), and the decode entry below shares its carried K/V
        // state by (name, aval, storage). Safe unconditionally: `self.cfg` is a `Qwen2Config`
        // with `head_dim = 64` (`TextConfig::smolvlm`), under both the flash-decode cap (`FLASH_LDS_CAP = 256`)
        // and flash-prefill cap (`FLASH_PREFILL_LDS_CAP = 512`); the text model is dense.
        // `trace_prefill_kv_embeds` shares its attention/MLP composition with `trace_prefill_kv`, which
        // `qwen2::tests::optimize_fuses_flash_prefill_no_materialized_scores` covers (see
        // `qwen2::tests::optimize_fuses_flash_prefill_vlm_embeds_no_materialized_scores` for the embeds path).
        let gp = trace_prefill_kv_embeds(self.cfg, n, cap);
        let mut pin: HashMap<ValueId, Value> = HashMap::new();
        for &id in &gp.inputs {
            let m = gp.meta(id);
            let t = match m.storage {
                Storage::Slot(Slot::Mask) => {
                    let name = m.name.as_deref().expect("mask slot without a name");
                    if name != "mask.prefill" {
                        bail!("unexpected mask slot {name} in VLM caption prefill");
                    }
                    causal_mask(n)
                }
                // Card 550: the mask is a graph computation over `Slot::Pos` and `iota`; this one-shot
                // prefill always starts at position 0.
                Storage::Slot(Slot::Pos) => crate::core::graphs::prefill_pos_rows(&m.aval, n),
                Storage::Slot(Slot::Activation) => {
                    let name = m.name.as_deref().expect("activation without a name");
                    if name != "activation.vlm.input_embeds" {
                        bail!("unexpected activation slot {name} in VLM caption prefill");
                    }
                    input_embeds.clone()
                }
                Storage::Const | Storage::State => continue,
                other => bail!("unexpected {other:?} input in VLM caption prefill"),
            };
            pin.insert(id, t.into());
        }
        let gp_staged = staged_program(&gp, executor_target(exec))
            .map_err(|e| err!("vlm prefill graph staging: {e}"))?;
        let gp_entry = exec
            .add_entry(exe, &gp_staged)
            .map_err(|e| err!("vlm prefill add_entry: {e}"))?;
        let prefill_result: Result<Vec<u8>> = (|| {
            let inputs = slot_step_inputs(&gp, &pin)?;
            exec.step(exe, gp_entry, &inputs, &mut poot_executor::NoSync)
                .map_err(|e| err!("prefill: {e}"))?
                .read()
                .map_err(|e| err!("prefill readback: {e}"))
        })();
        let logits_bytes = match exec.remove_entry(exe, gp_entry) {
            Ok(()) => prefill_result?,
            Err(e) if prefill_result.is_ok() => return Err(err!("vlm prefill remove_entry: {e}")),
            Err(_) => prefill_result?,
        };
        let mut logits: Vec<f32> = bytemuck::cast_slice(&logits_bytes).to_vec();

        // 3) greedy decode loop. Decode the cumulative output each step and emit the new suffix via `on_token`,
        // so a streaming caller gets pieces as produced (re-decoding the whole sequence avoids splitting a
        // multi-byte or multi-token grapheme; captions are short).
        //
        // The decode entry is staged once and replayed for every token; matches
        // `Runner::generate_kv_gpu_cached_sampled`'s qwen2/llama decode (`decode_kv_graph`), and shares the
        // prefill entry's carried state exactly as `generate_kv_gpu_prefilled` does. Always
        // removed before this returns, on every exit path (the loop's own `break`s and an early `?`).
        let gd = trace_decode_kv_masked(self.cfg, cap);
        let gd_staged = staged_program(&gd, executor_target(exec))
            .map_err(|e| err!("vlm decode graph staging: {e}"))?;
        let gd_entry = exec
            .add_entry(exe, &gd_staged)
            .map_err(|e| err!("vlm decode add_entry: {e}"))?;
        let decode_result: Result<String> = (|| {
            let mut out = Vec::new();
            let mut emitted = 0usize;
            for step in 0..max_new {
                let next = greedy_token(&logits)?;
                if next == self.eos {
                    break;
                }
                out.push(next);
                // `emitted` is a byte cursor into the re-decoded-each-step `s`, and `tokenizer.decode` is lossy UTF-8:
                // when the newest token only partly completes a multi-byte codepoint, `s` ends in U+FFFD replacement
                // chars of a different byte length than the final codepoint, so a raw `&s[emitted..]` could land off a
                // char boundary and panic (card 220). Use `stream_piece_delta`, as every other streaming path: `prev`
                // is the text already emitted (a valid boundary), `full` is the current decode, and `emitted` advances
                // only by the stable bytes emitted (never by `s.len()`, which may include a held-back U+FFFD
                // run).
                if let Ok(s) = self.tokenizer.decode(&out, true) {
                    let prev = s.get(..emitted).unwrap_or(&s);
                    let delta = stream_piece_delta(prev, &s);
                    if !delta.is_empty() {
                        if on_token(&delta).is_break() {
                            break;
                        }
                        emitted += delta.len();
                    }
                }
                let pos = n + step;
                let mut din: HashMap<ValueId, Value> = HashMap::new();
                for &id in &gd.inputs {
                    let m = gd.meta(id);
                    let t = match m.storage {
                        Storage::Slot(Slot::Token) => {
                            crate::core::graphs::token_slot(&m.aval, &[next])
                        }
                        // Card 550: `trace_decode_kv_masked` declares `Slot::Pos` as `[1,1]`, not a
                        // bare scalar; fill by element count so either declared shape binds the same
                        // single value. The mask is now a graph computation over `pos`, so the old
                        // `Slot::Mask`/`Slot::SeqLen` arms below are unreachable for this graph but
                        // stay in case a future caller passes an unconverted decode graph here.
                        Storage::Slot(Slot::Pos) => {
                            let n = m.aval.numel().max(1);
                            HostTensor::i32(m.aval.shape.clone(), vec![pos as i32; n])
                        }
                        Storage::Slot(Slot::SeqLen) => {
                            HostTensor::i32(vec![], vec![(pos + 1) as i32])
                        }
                        Storage::Slot(Slot::Mask) => HostTensor::f32(
                            vec![cap],
                            (0..cap)
                                .map(|t| if t <= pos { 0.0 } else { -1.0e9 })
                                .collect(),
                        ),
                        Storage::Const | Storage::State => continue,
                        _ => return Err(err!("unexpected decode input")),
                    };
                    din.insert(id, t.into());
                }
                let inputs = slot_step_inputs(&gd, &din)?;
                let bytes = exec
                    .step(exe, gd_entry, &inputs, &mut poot_executor::NoSync)
                    .map_err(|e| err!("decode: {e}"))?
                    .read()
                    .map_err(|e| err!("decode readback: {e}"))?;
                logits = bytemuck::cast_slice(&bytes).to_vec();
            }
            self.tokenizer
                .decode(&out, true)
                .map_err(|e| err!("decode caption: {e}"))
        })();
        match exec.remove_entry(exe, gd_entry) {
            Ok(()) => decode_result,
            Err(e) if decode_result.is_ok() => Err(err!("vlm decode remove_entry: {e}")),
            Err(_) => decode_result,
        }
    }
}

/// Builds the idefics3 image-placeholder block for a split image, a port of the reference
/// `Idefics3Processor.replace_image_token`. `rows`/`cols` are the tile grid (0,0 = unsplit, a single
/// global frame); `seq_len` is the per-frame visual-token count (`n_image_tokens`). For a split image the
/// layout is, per tile in row-major order, `<fake_token_around_image><row_R_col_C>{seq_len x <image>}`,
/// a newline after each row, then a trailing global frame
/// `\n<fake_token_around_image><global-img>{seq_len x <image>}<fake_token_around_image>`. The `<image>`
/// count is `(rows*cols + 1) * seq_len` (split) or `seq_len` (unsplit), the connector's total
/// visual-token count across all frames.
pub fn smolvlm_image_block(rows: usize, cols: usize, seq_len: usize) -> String {
    // The idefics3 tokenizer only registers `<row_R_col_C>` markers up to 6x6 (transformers
    // `processing_idefics3.py`), so a larger grid would emit markers that shred to byte pieces. SmolVLM's
    // geometry (longest edge 2048, tile 512) maxes at 4x4; this guards a future config that widens it.
    debug_assert!(
        rows <= 6 && cols <= 6,
        "tile grid {rows}x{cols} exceeds the 6x6 <row_R_col_C> markers the tokenizer registers"
    );
    let images = "<image>".repeat(seq_len);
    if rows == 0 && cols == 0 {
        return format!("<fake_token_around_image><global-img>{images}<fake_token_around_image>");
    }
    let mut s = String::new();
    for nh in 0..rows {
        for nw in 0..cols {
            s.push_str(&format!(
                "<fake_token_around_image><row_{}_col_{}>{images}",
                nh + 1,
                nw + 1
            ));
        }
        s.push('\n');
    }
    s.push_str(&format!(
        "\n<fake_token_around_image><global-img>{images}<fake_token_around_image>"
    ));
    s
}

/// The SmolVLM prompt for a chat turn with one or more images. Each image contributes its own
/// [`smolvlm_image_block`] (from its tile grid), concatenated in order before the question, as the
/// reference `Idefics3Processor` replaces each `<image>` placeholder with its image's block in order.
/// `grids` is one `(rows, cols)` per image (from [`poot_models::vision::idefics3_frame_geometry`]); the
/// total `<image>` count is `sum_i (rows_i*cols_i + 1) * seq_len`, matching the concatenated visual
/// tokens.
pub fn smolvlm_caption_prompt_multi(
    question: &str,
    grids: &[(usize, usize)],
    seq_len: usize,
) -> String {
    let blocks: String = grids
        .iter()
        .map(|&(rows, cols)| smolvlm_image_block(rows, cols, seq_len))
        .collect();
    format!("<|im_start|>User:{blocks}{question}<end_of_utterance>\nAssistant:")
}

/// Fetches a raw tensor by name with a clear error, materialized from the store (ADR-0103: the
/// store keeps bytes as stored; this decodes the one tensor a caller actually needs).
fn get_raw(st: &WeightStore, name: &str) -> Result<HostTensor> {
    materialize_f32(st, name).map_err(|e| err!("missing weight {name}: {e}"))
}

/// Loads the llama-class text sub-model from a VLM checkpoint (spec 049 slice 4c): reads
/// `model.text_model.*` (+ top-level `lm_head.weight`) and re-keys to the `model.*` names poot's
/// qwen2/llama tracer binds, with the HF `[out, in]` `*_proj.weight` linears transposed to `[in, out]`.
/// The embedding stays `[vocab, hidden]` (a gather source); the lm_head is transposed to
/// `[hidden, vocab]`.
pub fn load_text_weights(
    dir: impl AsRef<std::path::Path>,
    cfg: &TextConfig,
) -> Result<DenseWeights> {
    let st = safetensors::load_weight_store(dir).map_err(|e| err!("load safetensors: {e}"))?;
    let mut m = DenseWeights::new();
    let straight = |st: &WeightStore, name: &str| -> Result<HostTensor> { get_raw(st, name) };
    let t = "model.text_model";
    m.insert(
        "model.embed_tokens.weight".into(),
        straight(&st, &format!("{t}.embed_tokens.weight"))?,
    );
    for l in 0..cfg.layers {
        let src = format!("{t}.layers.{l}");
        let dst = format!("model.layers.{l}");
        m.insert(
            format!("{dst}.input_layernorm.weight"),
            straight(&st, &format!("{src}.input_layernorm.weight"))?,
        );
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            m.insert(
                format!("{dst}.self_attn.{proj}.weight"),
                transpose2d(&get_raw(&st, &format!("{src}.self_attn.{proj}.weight"))?),
            );
        }
        m.insert(
            format!("{dst}.post_attention_layernorm.weight"),
            straight(&st, &format!("{src}.post_attention_layernorm.weight"))?,
        );
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            m.insert(
                format!("{dst}.mlp.{proj}.weight"),
                transpose2d(&get_raw(&st, &format!("{src}.mlp.{proj}.weight"))?),
            );
        }
    }
    m.insert(
        "model.norm.weight".into(),
        straight(&st, &format!("{t}.norm.weight"))?,
    );
    // lm_head is a top-level tensor [vocab, hidden] -> transpose to [hidden, vocab] (the output matmul
    // layout). Keyed "lm_head.weight" to match the qwen2/llama tracer's output-projection constant.
    m.insert(
        "lm_head.weight".into(),
        transpose2d(&get_raw(&st, "lm_head.weight")?),
    );
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_eval::{EvalBudget, EvalOptions};

    /// The VLM greedy step returns the argmax for finite logits and the typed sampler fault for a NaN row (the
    /// old local maximum panicked in `partial_cmp().unwrap()`).
    #[test]
    fn greedy_token_is_a_sampler_fault_on_nan_logits() {
        assert_eq!(greedy_token(&[0.1, 3.0, 0.5]).unwrap(), 1);
        for logits in [[f32::NAN; 3], [0.1, f32::NAN, 3.0]] {
            match greedy_token(&logits) {
                Err(crate::RunnerError::Sampler(fault)) => assert!(
                    matches!(*fault, crate::SamplerFault::NonFiniteLogit { value, .. } if value.is_some_and(|v| v.is_nan())),
                    "expected a NaN fault, got {fault:?}"
                ),
                other => panic!("expected the typed sampler fault, got {other:?}"),
            }
        }
    }

    /// Pure logic test (no model/GPU) for the caption-streaming splice (card 220). The old
    /// `&s[emitted..]` slice panics when `emitted` lands mid-codepoint because the running decode's U+FFFD
    /// tail resolved into a real multi-byte char of a different length. Reproduces that shape (a 4-byte
    /// emoji built up one byte at a time via `String::from_utf8_lossy`, like `tokenizer.decode` over partial
    /// token bytes) and checks (1) the old raw-slice update does panic here, and (2) the
    /// `stream_piece_delta` update, as in `caption`'s loop, neither panics nor drops the emoji.
    #[test]
    fn caption_streaming_splice_does_not_panic_or_drop_char_on_split_codepoint() {
        let bytes: [u8; 4] = [0xF0, 0x9F, 0x98, 0x80]; // U+1F600, one grinning-face codepoint
        let mut buf: Vec<u8> = Vec::new();
        let steps: Vec<String> = bytes
            .iter()
            .map(|&b| {
                buf.push(b);
                String::from_utf8_lossy(&buf).to_string()
            })
            .collect();

        // (1) Old behavior: `on_token(&s[emitted..]); emitted = s.len();` panics once a held char boundary from a
        // shorter U+FFFD-only decode no longer lines up with the resolved codepoint.
        let panicked = std::panic::catch_unwind(|| {
            let mut emitted = 0usize;
            for s in &steps {
                if s.len() > emitted {
                    let _piece = &s[emitted..]; // the old raw slice (card 220 Bug B)
                    emitted = s.len();
                }
            }
        })
        .is_err();
        assert!(
            panicked,
            "sanity: the OLD raw &s[emitted..] slice must panic on this split-codepoint input \
             (if this fails, the repro no longer matches the bug)"
        );

        // (2) New behavior (as the loop in `caption`): boundary-safe, reassembles the whole codepoint exactly
        // once via `stream_piece_delta`.
        let mut emitted = 0usize;
        let mut collected = String::new();
        for s in &steps {
            let prev = s.get(..emitted).unwrap_or(s);
            let delta = stream_piece_delta(prev, s);
            if !delta.is_empty() {
                collected.push_str(&delta);
                emitted += delta.len();
            }
        }
        assert_eq!(
            collected, "\u{1F600}",
            "the emoji must be reassembled whole, with no premature/placeholder emission"
        );
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint + GPU (multi-frame idefics3 split -> 17 encoder runs)"]
    fn vlm_runner_captions_an_image() {
        // Spec 049 slice 4c: `VlmRunner` captions an image on the real smolvlm-256m on the Arc GPU. With
        // idefics3 splitting the 96x96 input becomes 17 frames (4x4 tiles + global), exercising the
        // multi-frame vision + splice.
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("smolvlm-256m"))
        else {
            return;
        };
        let device = match poot_gpu::device::WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let vlm = VlmRunner::load(&dir).unwrap();
        assert_eq!(vlm.n_image_tokens(), 64);
        let mut engine = poot_executor::Engine::new(device);
        let exe = vlm.load_on(&mut engine).unwrap();
        let (ih, iw) = (96usize, 96usize);
        let rgb: Vec<f32> = (0..ih * iw * 3).map(|i| ((i * 7) % 256) as f32).collect();
        let caption = vlm
            .caption(
                &rgb,
                ih,
                iw,
                "Can you describe this image?",
                12,
                &mut engine,
                exe,
            )
            .unwrap();
        eprintln!("VlmRunner caption: {caption:?}");
        assert!(!caption.trim().is_empty(), "produced an empty caption");
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint + GPU (bails in preprocessing, before any dispatch)"]
    fn vlm_rejects_too_many_images_instead_of_ooming() {
        // DoS guard: each image tiles into ~17 in-memory frames and the image count is client-controlled.
        // `caption_streaming_multi` caps the total frames (MAX_VLM_FRAMES) and bails in the preprocessing loop,
        // before the vision tower runs (no GPU dispatch, so the test is fast).
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("smolvlm-256m"))
        else {
            return;
        };
        let device = match poot_gpu::device::WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let vlm = VlmRunner::load(&dir).unwrap();
        let mut engine = poot_executor::Engine::new(device);
        let exe = vlm.load_on(&mut engine).unwrap();
        // 40 tiny images; each tiles to ~17 frames, so ~680 total > MAX_VLM_FRAMES (512).
        let one: Vec<f32> = (0..4 * 4 * 3).map(|i| (i % 256) as f32).collect();
        let owned: Vec<(Vec<f32>, usize, usize)> =
            (0..40).map(|_| (one.clone(), 4usize, 4usize)).collect();
        let images: Vec<(&[f32], usize, usize)> = owned
            .iter()
            .map(|(r, h, w)| (r.as_slice(), *h, *w))
            .collect();
        let err = vlm
            .caption_streaming_multi(&images, "describe", 8, &mut engine, exe, |_| {
                std::ops::ControlFlow::Continue(())
            })
            .expect_err("must reject an oversized image batch");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("too many image frames"),
            "expected a frame-cap error, got: {msg}"
        );
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint + GPU (two captions on one executor, ~3 min)"]
    fn vlm_two_captions_on_one_executor_are_not_stale() {
        // Regression for the old const-cache staleness bug: the poot-serve VLM thread reuses one GpuExecutor,
        // and before card 550a the prefill's `vlm.input_embeds` + `causal.mask` were consts cached by name
        // with only an element-count staleness check, so two same-length captions (e.g. the same 12-token
        // question) could reuse the first image's embeddings. Those inputs are step inputs now, so the cache
        // cannot serve them at all; this test pins the behavior: captions two very different images (solid
        // gray vs high-contrast stripes) on one runner+executor and asserts the captions differ.
        // [[gpu-const-cache-staleness]]
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("smolvlm-256m"))
        else {
            return;
        };
        let device = match poot_gpu::device::WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let vlm = VlmRunner::load(&dir).unwrap();
        let mut engine = poot_executor::Engine::new(device);
        let exe = vlm.load_on(&mut engine).unwrap();
        let (ih, iw) = (96usize, 96usize);
        let solid: Vec<f32> = vec![128.0; ih * iw * 3];
        let striped: Vec<f32> = (0..ih * iw)
            .flat_map(|i| {
                let v = if (i / iw / 8) % 2 == 0 { 0.0 } else { 255.0 };
                [v, v, v]
            })
            .collect();
        let q = "Can you describe this image?";
        let cap_solid = vlm
            .caption(&solid, ih, iw, q, 12, &mut engine, exe)
            .unwrap();
        // second caption on the SAME executor: must reflect the striped image, not reuse the solid one.
        let cap_striped = vlm
            .caption(&striped, ih, iw, q, 12, &mut engine, exe)
            .unwrap();
        eprintln!("solid:   {cap_solid:?}");
        eprintln!("striped: {cap_striped:?}");
        assert!(!cap_solid.trim().is_empty() && !cap_striped.trim().is_empty());
        assert_ne!(
            cap_solid, cap_striped,
            "the second caption reused the first image's cached embeddings (const-cache staleness)"
        );
    }

    #[test]
    #[ignore = "needs smolvlm-256m + GPU (two images in one turn -> 34 encoder runs, ~4 min)"]
    fn vlm_captions_two_images_in_one_turn() {
        // Multi-image chat: two distinct images in one turn caption together via `caption_streaming_multi`.
        // Each 96x96 image splits into 17 frames, so this runs 34 vision-encoder passes and a ~2200-token
        // prefill (2*1088 image tokens + text), exercising multi-image preprocess -> concat -> multi-block
        // prompt -> splice end to end.
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("smolvlm-256m"))
        else {
            return;
        };
        let device = match poot_gpu::device::WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let vlm = VlmRunner::load(&dir).unwrap();
        let mut engine = poot_executor::Engine::new(device);
        let exe = vlm.load_on(&mut engine).unwrap();
        let (ih, iw) = (96usize, 96usize);
        let solid: Vec<f32> = vec![128.0; ih * iw * 3];
        let striped: Vec<f32> = (0..ih * iw)
            .flat_map(|i| {
                let v = if (i / iw / 8) % 2 == 0 { 0.0 } else { 255.0 };
                [v, v, v]
            })
            .collect();
        let caption = vlm
            .caption_streaming_multi(
                &[(&solid, ih, iw), (&striped, ih, iw)],
                "Describe both images.",
                16,
                &mut engine,
                exe,
                |_| std::ops::ControlFlow::Continue(()),
            )
            .unwrap();
        eprintln!("two-image caption: {caption:?}");
        assert!(!caption.trim().is_empty(), "produced an empty caption");
    }

    #[test]
    fn decode_image_round_trips_png_and_data_url() {
        use base64::Engine;
        // build a 2x3 RGB image with known pixels, encode to PNG, and decode back.
        let (w, h) = (2u32, 3u32);
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(x, y, image::Rgb([(x * 100) as u8, (y * 50) as u8, 200]));
            }
        }
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let bytes = png.into_inner();

        let (rgb, dh, dw) = decode_image(&bytes).unwrap();
        assert_eq!((dh, dw), (h as usize, w as usize));
        assert_eq!(rgb.len(), (h * w * 3) as usize);
        // pixel (x=1, y=2): R=100, G=100, B=200.
        let p = (2 * w as usize + 1) * 3;
        assert_eq!(&rgb[p..p + 3], &[100.0, 100.0, 200.0]);

        // the same bytes as a data URL round-trip identically.
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let url = format!("data:image/png;base64,{b64}");
        let (rgb2, _, _) = decode_image_data_url(&url).unwrap();
        assert_eq!(rgb, rgb2);
    }

    #[test]
    fn check_image_dims_rejects_decompression_bombs() {
        assert!(check_image_dims(64, 64).is_ok());
        assert!(check_image_dims(1024, 1024).is_ok());
        // exactly at the cap (8192 * 4096 == 32 * 1024 * 1024) is allowed.
        assert!(check_image_dims(8192, 4096).is_ok());
        // a decompression bomb (900 MP) is rejected with a clear message.
        let e = check_image_dims(30000, 30000).unwrap_err().to_string();
        assert!(e.contains("exceeds"), "unexpected message: {e}");
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m tokenizer"]
    fn caption_prompt_tokenizes_to_matching_image_tokens() {
        // Spec 049 slice 4c: the caption prompt tokenizes so the number of <image> (49190) ids equals the
        // connector's visual-token count; the splice's inv map needs one image position per visual token.
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("smolvlm-256m"))
        else {
            return;
        };
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let n_vis = 64usize;
        let prompt = smolvlm_caption_prompt_multi("Can you describe this image?", &[(0, 0)], n_vis);
        let ids = tok.encode(prompt, false).unwrap().get_ids().to_vec();
        let image_tokens = ids.iter().filter(|&&t| t == 49190).count();
        assert_eq!(
            image_tokens, n_vis,
            "image-token count must match the visual tokens"
        );
        // the structural special tokens are present (global image + the fake-token wrappers).
        assert!(ids.contains(&49152), "<global-img>");
        assert!(ids.contains(&49189), "<fake_token_around_image>");
    }

    #[test]
    fn tiled_caption_prompt_has_correct_image_token_count_and_markers() {
        // GPU-free: the split-image prompt (idefics3 replace_image_token) emits (rows*cols + 1) * seq_len
        // <image> placeholders (one per visual token across all frames + the global) plus one <row_R_col_C>
        // marker per tile in row-major order, and ends before the assistant turn.
        let seq = 64usize;
        let (rows, cols) = (2usize, 4usize); // e.g. a 512x1024 landscape (9 frames).
        let p = smolvlm_caption_prompt_multi("Describe.", &[(rows, cols)], seq);
        let images = p.matches("<image>").count();
        assert_eq!(
            images,
            (rows * cols + 1) * seq,
            "one <image> per visual token"
        );
        // every tile marker present exactly once, row-major.
        for nh in 1..=rows {
            for nw in 1..=cols {
                let m = format!("<row_{nh}_col_{nw}>");
                assert_eq!(p.matches(&m).count(), 1, "missing/dup {m}");
            }
        }
        assert_eq!(p.matches("<global-img>").count(), 1, "one global frame");
        assert!(p.ends_with("<end_of_utterance>\nAssistant:"));

        // the unsplit (0,0) case is a single global frame with no tile markers.
        let single = smolvlm_caption_prompt_multi("Describe.", &[(0, 0)], seq);
        assert_eq!(single.matches("<image>").count(), seq);
        assert_eq!(
            single.matches("<row_").count(),
            0,
            "no tile markers when unsplit"
        );
    }

    #[test]
    fn preprocess_and_prompt_agree_on_image_token_count() {
        // VLM integration invariant: the idefics3 split geometry (frames per image) and the tiled prompt
        // (<image> placeholders) must agree, or the splice's inverse map has the wrong number of image
        // positions and the caption is corrupt. Checked across realistic image sizes (h, w): the prompt's
        // <image> count equals n_frames * n_vis, and the tile grid stays within the tokenizer's 6x6 markers.
        use poot_models::vision::{PreprocessConfig, idefics3_frame_geometry};
        let cfg = PreprocessConfig::smolvlm();
        let n_vis = 64usize; // smolvlm connector output per frame (num_patches/scale^2 = 1024/16).
        let sizes = [
            (64usize, 64usize),
            (512, 512),
            (96, 96),
            (800, 600),
            (600, 800),
            (1920, 1080),
            (1080, 1920),
            (100, 2000),
            (2000, 100),
            (1, 1),
            (3, 4000),
        ];
        for (h, w) in sizes {
            let (rows, cols, n_frames) = idefics3_frame_geometry(h, w, &cfg);
            assert!(
                rows <= 6 && cols <= 6,
                "{h}x{w} -> {rows}x{cols} exceeds the tokenizer's 6x6 <row_R_col_C> markers"
            );
            let prompt = smolvlm_caption_prompt_multi("Describe.", &[(rows, cols)], n_vis);
            let images = prompt.matches("<image>").count();
            assert_eq!(
                images,
                n_frames * n_vis,
                "{h}x{w}: prompt <image> count {images} != n_frames {n_frames} * n_vis {n_vis}"
            );
        }
    }

    #[test]
    fn multi_image_prompt_token_count_sums_across_images() {
        // Multi-image chat: one <image> block per image (each from its own tile grid), so the total <image>
        // count is the sum of per-image frame counts * n_vis, which must equal the concatenated visual-token
        // count across all images' frames (the multi-image splice invariant).
        use poot_models::vision::{PreprocessConfig, idefics3_frame_geometry};
        let cfg = PreprocessConfig::smolvlm();
        let n_vis = 64usize;
        let sizes = [(96usize, 96usize), (512, 1024), (600, 800)];
        let mut grids = Vec::new();
        let mut expected = 0usize;
        for (h, w) in sizes {
            let (r, c, nf) = idefics3_frame_geometry(h, w, &cfg);
            grids.push((r, c));
            expected += nf * n_vis;
        }
        let prompt = smolvlm_caption_prompt_multi("Compare these.", &grids, n_vis);
        assert_eq!(
            prompt.matches("<image>").count(),
            expected,
            "total <image> count must sum each image's frames * n_vis"
        );
        // each image contributes exactly one global thumbnail block.
        assert_eq!(prompt.matches("<global-img>").count(), grids.len());
        assert!(prompt.ends_with("<end_of_utterance>\nAssistant:"));
        // the single-image prompt is the chat wrapper around that image's own block.
        assert_eq!(
            smolvlm_caption_prompt_multi("Q", &[(2, 3)], n_vis),
            format!(
                "<|im_start|>User:{}Q<end_of_utterance>\nAssistant:",
                smolvlm_image_block(2, 3, n_vis)
            )
        );
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint (~60s CPU)"]
    fn real_text_model_generates_coherent_text() {
        // Spec 049 slice 4c: the full generation loop (prefill -> greedy decode -> detokenize) on the real
        // smolvlm text model yields a coherent multi-token continuation. This is the loop image captioning
        // uses (with `trace_prefill_kv_embeds` and the spliced image embeds for the prefill).

        use poot_graph_ir::{Slot, Storage, ValueId};
        use poot_models::qwen2::{Qwen2Config, trace_decode_kv_masked, trace_prefill_kv};
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("smolvlm-256m"))
        else {
            return;
        };
        let tcfg = TextConfig::smolvlm();
        let cfg = Qwen2Config {
            vocab: tcfg.vocab,
            hidden: tcfg.hidden,
            inter: tcfg.intermediate,
            layers: tcfg.layers,
            n_heads: 9,
            n_kv_heads: 3,
            head_dim: 64,
            rotary_dim: 64,
            eps: 1e-5,
            max_pos: 8192,
            qkv_bias: false,
            qk_norm: false,
            ..Default::default()
        };
        let wmap = load_text_weights(&dir, &tcfg).unwrap();
        let (cos, sin) = rope_tables(&cfg, 100000.0, None, None);
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        // bind any const by name (weights + the rope tables).
        let cst = |name: &str| -> HostTensor {
            match name {
                "rope.cos" => cos.clone(),
                "rope.sin" => sin.clone(),
                n => wmap[n].clone(),
            }
        };

        let prompt = "The capital of France is";
        let ids: Vec<u32> = tok.encode(prompt, false).unwrap().get_ids().to_vec();
        let n = ids.len();
        let max_new = 6usize;
        let cap = n + max_new;

        // prefill.
        let gp = trace_prefill_kv(cfg, n, cap);
        let mut pin: HashMap<ValueId, Value> = HashMap::new();
        for &id in &gp.inputs {
            let m = gp.meta(id);
            match m.storage {
                Storage::Slot(Slot::Token) => {
                    pin.insert(
                        id,
                        HostTensor::i32(vec![n], ids.iter().map(|&t| t as i32).collect()).into(),
                    );
                }
                Storage::Slot(Slot::Mask) => {
                    let name = m.name.as_deref().unwrap();
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    pin.insert(id, causal_mask(n).into());
                }
                // Card 550: the mask is a graph computation over `Slot::Pos` and `iota`; this one-shot
                // prefill always starts at position 0.
                Storage::Slot(Slot::Pos) => {
                    pin.insert(id, crate::core::graphs::prefill_pos_rows(&m.aval, n).into());
                }
                Storage::Const => {
                    let name = m.name.as_deref().unwrap();
                    pin.insert(id, cst(name).into());
                }
                Storage::State => {}
                _ => panic!("prefill {:?}", m.storage),
            }
        }
        for &(si, _) in &gp.state {
            pin.insert(si, HostTensor::zeros(gp.aval(si).shape.clone()).into());
        }
        let prefill_step =
            poot_eval::eval(&gp, &pin, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
        let mut logits = prefill_step.output.into_host().expect("dense output");
        let mut caches: Vec<HostTensor> = prefill_step
            .state
            .into_iter()
            .map(|v| v.into_host().expect("dense state"))
            .collect();

        // greedy decode loop.
        let gd = trace_decode_kv_masked(cfg, cap);
        let mut out = Vec::new();
        for step in 0..max_new {
            let next = greedy_token(logits.as_f32().unwrap()).unwrap();
            out.push(next);
            if step + 1 == max_new {
                break;
            }
            let pos = n + step;
            let mut din: HashMap<ValueId, Value> = HashMap::new();
            for &id in &gd.inputs {
                let m = gd.meta(id);
                let t = match m.storage {
                    Storage::Slot(Slot::Token) => HostTensor::i32(vec![], vec![next as i32]),
                    // Card 550: `Slot::Pos` is now `[1,1]`, not a bare scalar; fill by element count.
                    Storage::Slot(Slot::Pos) => {
                        let elems = m.aval.numel().max(1);
                        HostTensor::i32(m.aval.shape.clone(), vec![pos as i32; elems])
                    }
                    Storage::Slot(Slot::SeqLen) => HostTensor::i32(vec![], vec![(pos + 1) as i32]),
                    Storage::Slot(Slot::Mask) => HostTensor::f32(
                        vec![cap],
                        (0..cap)
                            .map(|t| if t <= pos { 0.0 } else { -1.0e9 })
                            .collect(),
                    ),
                    Storage::Const => cst(m.name.as_deref().unwrap()),
                    Storage::State => continue,
                    _ => panic!("decode {:?}", m.storage),
                };
                din.insert(id, t.into());
            }
            for (ci, &(si, _)) in gd.state.iter().enumerate() {
                din.insert(si, caches[ci].clone().into());
            }
            let step_eval =
                poot_eval::eval(&gd, &din, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
            logits = step_eval.output.into_host().expect("dense output");
            caches = step_eval
                .state
                .into_iter()
                .map(|v| v.into_host().expect("dense state"))
                .collect();
        }
        let text = tok.decode(&out, false).unwrap();
        eprintln!("prompt {prompt:?} -> generated {out:?} = {text:?}");
        assert!(
            text.to_lowercase().contains("paris"),
            "expected a coherent continuation, got {text:?}"
        );
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint (~25s CPU)"]
    fn real_text_model_predicts_sensible_next_token() {
        // Spec 049 slice 4c: the real smolvlm text model, given a factual prompt, greedily predicts a sensible
        // next token, proving the text weights and decode are correct (not just finite); this needs every
        // transpose/name/rope to be right.

        use poot_graph_ir::{Slot, Storage, ValueId};
        use poot_models::qwen2::{Qwen2Config, trace_prefill_kv};
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("smolvlm-256m"))
        else {
            return;
        };
        let tcfg = TextConfig::smolvlm();
        let cfg = Qwen2Config {
            vocab: tcfg.vocab,
            hidden: tcfg.hidden,
            inter: tcfg.intermediate,
            layers: tcfg.layers,
            n_heads: 9,
            n_kv_heads: 3,
            head_dim: 64,
            rotary_dim: 64,
            eps: 1e-5,
            max_pos: 8192,
            qkv_bias: false,
            qk_norm: false,
            ..Default::default()
        };
        let wmap = load_text_weights(&dir, &tcfg).unwrap();
        let (cos, sin) = rope_tables(&cfg, 100000.0, None, None);
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();

        let prompt = "The capital of France is";
        let ids: Vec<u32> = tok.encode(prompt, false).unwrap().get_ids().to_vec();
        let n = ids.len();
        let cap = n;

        let g = trace_prefill_kv(cfg, n, cap);
        let mut inputs: HashMap<ValueId, Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let t = match meta.storage {
                Storage::Slot(Slot::Token) => {
                    HostTensor::f32(vec![n], ids.iter().map(|&t| t as f32).collect())
                }
                Storage::Slot(Slot::Mask) => {
                    let name = meta.name.as_deref().unwrap();
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    causal_mask(n)
                }
                Storage::Const => match meta.name.as_deref().unwrap() {
                    "rope.cos" => cos.clone(),
                    "rope.sin" => sin.clone(),
                    name => wmap[name].clone(),
                },
                Storage::State => continue,
                _ => panic!("unexpected {:?}", meta.storage),
            };
            inputs.insert(id, t.into());
        }
        for &(si, _) in &g.state {
            inputs.insert(si, HostTensor::zeros(g.aval(si).shape.clone()).into());
        }
        let logits = poot_eval::eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("dense output");
        let next = logits
            .as_f32()
            .unwrap()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        let word = tok.decode(&[next], false).unwrap();
        eprintln!("prompt {prompt:?} -> next token {next} = {word:?}");
        assert!(logits.as_f32().unwrap().iter().all(|v| v.is_finite()));
        // SmolLM2 knows this fact: the greedy continuation is "Paris".
        assert!(
            word.to_lowercase().contains("paris"),
            "expected a Paris-ish continuation, got {word:?}"
        );
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint (~20s CPU)"]
    fn real_text_model_prefill_from_embeds_matches_tokens() {
        // Spec 049 slice 4c: the real smolvlm text model (30-layer llama) prefills via the embeddings path
        // identically to the token-id path, confirming `load_text_weights` yields correctly shaped/keyed tensors
        // that the qwen2/llama tracer binds and runs (finite 30-layer prefill).

        use poot_graph_ir::{Slot, Storage, ValueId};
        use poot_models::qwen2::{Qwen2Config, trace_prefill_kv, trace_prefill_kv_embeds};
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("smolvlm-256m"))
        else {
            return;
        };
        let tcfg = TextConfig::smolvlm();
        let cfg = Qwen2Config {
            vocab: tcfg.vocab,
            hidden: tcfg.hidden,
            inter: tcfg.intermediate,
            layers: tcfg.layers,
            n_heads: 9,
            n_kv_heads: 3,
            head_dim: 64,
            rotary_dim: 64,
            eps: 1e-5,
            max_pos: 8192,
            qkv_bias: false,
            qk_norm: false,
            ..Default::default()
        };
        let wmap = load_text_weights(&dir, &tcfg).unwrap();
        let (cos, sin) = rope_tables(&cfg, 100000.0, None, None);
        let tokens: Vec<u32> = vec![1, 100, 250, 42];
        let n = tokens.len();
        let cap = n;
        let h = cfg.hidden;

        // input embeds = the real embedding rows at `tokens`.
        let embed = &wmap["model.embed_tokens.weight"];
        let mut input_embeds = vec![0.0f32; n * h];
        for (i, &t) in tokens.iter().enumerate() {
            input_embeds[i * h..(i + 1) * h]
                .copy_from_slice(&embed.as_f32().unwrap()[t as usize * h..(t as usize + 1) * h]);
        }

        let bind = |g: &poot_graph_ir::Graph, embeds: Option<&[f32]>| -> HashMap<ValueId, Value> {
            let mut m = HashMap::new();
            for &id in &g.inputs {
                let meta = g.meta(id);
                let t = match meta.storage {
                    Storage::Slot(Slot::Token) => {
                        HostTensor::f32(vec![n], tokens.iter().map(|&t| t as f32).collect())
                    }
                    Storage::Slot(Slot::Mask) => {
                        let name = meta.name.as_deref().unwrap();
                        assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                        causal_mask(n)
                    }
                    Storage::Slot(Slot::Activation) => {
                        let name = meta.name.as_deref().unwrap();
                        assert_eq!(
                            name, "activation.vlm.input_embeds",
                            "unexpected activation slot {name}"
                        );
                        HostTensor::f32(vec![n, h], embeds.unwrap().to_vec())
                    }
                    Storage::Const => {
                        let name = meta.name.as_deref().unwrap();
                        match name {
                            "rope.cos" => cos.clone(),
                            "rope.sin" => sin.clone(),
                            _ => wmap[name].clone(),
                        }
                    }
                    Storage::State => continue,
                    _ => panic!("unexpected {:?}", meta.storage),
                };
                m.insert(id, t.into());
            }
            for &(si, _) in &g.state {
                m.insert(si, HostTensor::zeros(g.aval(si).shape.clone()).into());
            }
            m
        };

        let gt = trace_prefill_kv(cfg, n, cap);
        let lt = poot_eval::eval(
            &gt,
            &bind(&gt, None),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap()
        .output
        .into_host()
        .expect("dense output");
        let ge = trace_prefill_kv_embeds(cfg, n, cap);
        let le = poot_eval::eval(
            &ge,
            &bind(&ge, Some(&input_embeds)),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap()
        .output
        .into_host()
        .expect("dense output");

        assert!(
            lt.as_f32().unwrap().iter().all(|v| v.is_finite()),
            "real text prefill not finite"
        );
        assert_eq!(lt.as_f32().unwrap().len(), le.as_f32().unwrap().len());
        for (i, (a, b)) in lt
            .as_f32()
            .unwrap()
            .iter()
            .zip(le.as_f32().unwrap())
            .enumerate()
        {
            assert!(
                (a - b).abs() <= 1e-4 * a.abs().max(1.0),
                "logit {i}: tok {a} vs emb {b}"
            );
        }
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint"]
    fn load_text_weights_shapes() {
        // Spec 049 slice 4c: the llama-class text sub-model loads, re-keyed to poot's `model.*` names with proj
        // linears transposed to [in, out], so the qwen2/llama tracer can bind it.
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("smolvlm-256m"))
        else {
            return;
        };
        let cfg = TextConfig::smolvlm();
        let m = load_text_weights(&dir, &cfg).unwrap();
        assert_eq!(
            m["model.embed_tokens.weight"].shape(),
            vec![cfg.vocab, cfg.hidden]
        );
        assert_eq!(
            m["model.layers.0.self_attn.q_proj.weight"].shape(),
            vec![cfg.hidden, cfg.q_dim],
            "q_proj transposed to [in, out]"
        );
        assert_eq!(
            m["model.layers.0.self_attn.k_proj.weight"].shape(),
            vec![cfg.hidden, cfg.kv_dim]
        );
        assert_eq!(
            m["model.layers.0.mlp.down_proj.weight"].shape(),
            vec![cfg.intermediate, cfg.hidden]
        );
        assert_eq!(
            m["model.layers.0.input_layernorm.weight"].shape(),
            vec![cfg.hidden]
        );
        assert_eq!(m["model.norm.weight"].shape(), vec![cfg.hidden]);
        assert_eq!(
            m["lm_head.weight"].shape(),
            vec![cfg.hidden, cfg.vocab],
            "lm_head [hidden, vocab]"
        );
        // count: embed + norm + lm_head + 9 per layer (1 ln_in + 4 attn + 1 ln_post + 3 mlp).
        assert_eq!(m.len(), 3 + 9 * cfg.layers);
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint"]
    fn load_vision_weights_shapes() {
        // Spec 049 slice 4b: the real SigLIP vision weights load with the expected shapes (incl. the conv-weight
        // transpose to the linear [C*P*P, H] layout), keyed by the graph's constant names.
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("smolvlm-256m"))
        else {
            return;
        };
        let cfg = VisionConfig::smolvlm();
        let m = load_vision_weights(&dir, &cfg).unwrap();
        let h = cfg.hidden();
        let pd = cfg.channels * cfg.patch_size * cfg.patch_size;
        assert_eq!(m["patch_w"].shape(), vec![pd, h], "patch_w [C*P*P, H]");
        assert_eq!(m["patch_b"].shape(), vec![h]);
        assert_eq!(m["pos_embed"].shape(), vec![cfg.num_patches(), h]);
        assert_eq!(
            m["wq0"].shape(),
            vec![h, h],
            "q_proj transposed to [in,out]"
        );
        assert_eq!(m["f1w0"].shape(), vec![h, 3072], "fc1 [H, intermediate]");
        assert_eq!(m["f2w0"].shape(), vec![3072, h], "fc2 [intermediate, H]");
        assert_eq!(m["plw"].shape(), vec![h]);
        // every value finite.
        assert!(m["patch_w"].as_f32().unwrap().iter().all(|v| v.is_finite()));
        // count: 3 (patch_w/b, pos) + 16/layer + 2 (post-ln).
        assert_eq!(m.len(), 3 + 16 * cfg.layers + 2);
    }
}
