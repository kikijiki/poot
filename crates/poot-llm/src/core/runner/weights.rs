use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use poot_eval::{Value, take_dense};
use poot_graph_ir::{
    Eqn, Graph, OpKind, Operand, PackedSourceName, Storage, TensorType, ValueId, ValueMeta,
};
use poot_graph_plan::{PackedLayout, WeightFormats};
use poot_load::Qwen2HfConfig;
use poot_models::qwen2::Qwen2Config;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::{DType, HostTensor};

use super::yarn::deepseek2_yarn_params;
use crate::checkpoint::gguf::permute::gather_from;
use crate::checkpoint::gguf::{rope::rope_tables, transpose_experts, transpose2d};
use crate::checkpoint::place::place_packed;
use crate::error::{OptionExt, Result, ResultExt};

/// The weight map of a loader whose every weight is a dense tensor (an unquantized architecture's
/// builder): each entry a `Value::Host`, built once at load. A loader with packed weights inserts
/// its `Value::Packed` sources itself.
pub(crate) fn dense_weight_map(dense: HashMap<String, HostTensor>) -> HashMap<String, Value> {
    dense
        .into_iter()
        .map(|(name, tensor)| (name, Value::Host(tensor)))
        .collect()
}

/// Inserts `name` into `w`, refusing a second write of the same name as a typed hard error instead
/// of a silent `HashMap::insert` overwrite (card 540b review: a tied checkpoint's redundant on-disk
/// `lm_head.weight` clobbering the correct embedding-derived value was exactly this failure mode,
/// and it cost a real regression). Every insert `build_weights` makes goes through this - not a bare
/// `.insert()` - so any future ordering gap that lets two sources write the same name fails loudly at
/// the source checkpoint's own load time instead of quietly keeping whichever source ran last.
fn insert_once(w: &mut HashMap<String, Value>, name: String, t: impl Into<Value>) -> Result<()> {
    if w.insert(name.clone(), t.into()).is_some() {
        bail!(
            "build_weights: {name} was written twice (two sources produced the same weight name)"
        );
    }
    Ok(())
}

/// Builds the runner's weight map (the name -> host tensor map the prefill graph binds) from a
/// checkpoint's [`WeightStore`] (card 540b: bytes as stored, no host f32 widening at load time;
/// this function is where each tensor this arch's tracer needs gets materialized, each in its
/// STORED dtype: a bf16/f16 checkpoint weight stays its 16-bit words). Linear `*proj.weight`
/// tensors are `[out,in]` in the checkpoint and transposed to `[in,out]`; the embedding stays
/// `[vocab,hidden]` (a gather source); the tied lm_head is the embedding transposed; RoPE cos/sin
/// tables are computed. A GPTQ/AWQ/FP8 checkpoint's quantized linears stay packed (no dense copy is
/// materialized).
///
/// MoE checkpoints (qwen3-moe/Mixtral/OlmoE/DeepSeek-V2) never materialize their raw per-expert tensors
/// into `w`: every family-specific block below (`is_qwen3_moe`/`is_mixtral`/`is_olmoe`/`is_deepseek2`/
/// `is_deepseek3`) drains its own per-expert tensors via `fuse_qwen3_moe_experts`/`fuse_mixtral_experts`
/// *before* the generic per-tensor loop below ever lists `store`'s remaining keys, so the raw copies are
/// simply gone from the store by the time that loop runs - no name-based skip needed. See
/// `docs/updates/` (search "cpu-eager memory") for the DeepSeek-V2-Lite peak-memory measurement this
/// fixed (a ~186GB transient peak from the raw-expert tensors being fully transposed and bf16-cloned
/// into `w` while the same data was still resident in `store`).
///
/// This "family blocks drain their own names first, then the generic loop takes whatever is left" order
/// (card 540b review) is why every family-specific block below runs *before* the generic loop rather than
/// after it: `take_dense` removes an entry on first read (draining the store as `build_weights` consumes
/// it, card 540b SC-001), so a name a family block will read by its own exact string - the untied
/// `lm_head.weight`, DeepSeek's `kv_a_proj_with_mqa.weight`, every router, granite's per-expert 3D
/// tensors - must be gone from the store *before* the generic loop lists `store.keys()`, or the generic
/// loop takes it first as a plain untransposed pass-through and the family block's later read finds
/// nothing. Under the old borrowing `materialize_dense` this ordering did not matter (a second read just
/// re-read the same bytes); a hard-coded list of family tensor-name suffixes in the generic loop was tried
/// first to plug the resulting "missing tensor" regression and rejected in review as exactly the kind of
/// family-specific knowledge that does not belong in generic code - the loop needs no name list at all
/// once it genuinely only ever sees names nothing else wants.
pub(crate) fn build_weights(
    store: &mut WeightStore,
    cfg: &Qwen2Config,
    hf: &Qwen2HfConfig,
) -> Result<(HashMap<String, Value>, WeightFormats)> {
    let mut w = HashMap::new();
    let mut formats = WeightFormats::default();

    // The embedding table is materialized once, up front, and feeds both `model.embed_tokens.weight`
    // (untransposed, a gather source) and, for a tied model (the common case: no separate on-disk
    // `lm_head.weight`), the transposed `lm_head.weight`. A second `materialize_dense` for the same
    // tensor later (the naive "handle lm_head after the generic loop" structure) decoded a
    // multi-hundred-MB embedding table twice, a real peak-RSS cost measured against the M4 gate
    // (card 540b SC-001): the store (resident for this whole function) plus two live decodes of the
    // checkpoint's largest tensor at once.
    let tied = hf.tie_word_embeddings || !store.contains("lm_head.weight");
    let embed = take_dense(store, "model.embed_tokens.weight").context("missing embed_tokens")?;
    if tied {
        insert_once(&mut w, "lm_head.weight".to_string(), transpose2d(&embed))?;
    }
    insert_once(&mut w, "model.embed_tokens.weight".to_string(), embed)?;
    // Output projection -> [hidden, vocab] matmul layout. The tied case (the common one) was already
    // inserted above, from the same materialized embedding table. An untied model (e.g. mistral) ships
    // a separate on-disk `lm_head.weight` [vocab, hidden]; take and transpose it here.
    //
    // Drain it unconditionally whenever the store holds one, tied or not (card 540b review: a
    // confirmed regression). Some tied checkpoints (qwen2.5-0.5b-awq, -fp8) still ship a redundant
    // on-disk `lm_head.weight` even with `tie_word_embeddings` set - a quantization tool's conversion
    // artifact, not a hypothetical. Left in the store, that entry is not caught by the "family blocks
    // drain their own names first" ordering above (nothing else claims this name in the tied case), so
    // the generic loop below takes it as a plain untransposed `[vocab, hidden]` pass-through and
    // silently overwrote the correct transposed `[hidden, vocab]` value the tied branch just inserted
    // (observed as garbled AWQ resident-GPU generation, `awq_safetensors_resident_packed_decode_on_gpu`).
    if store.contains("lm_head.weight") {
        let lm = take_dense(store, "lm_head.weight").context("lm_head.weight")?;
        if !tied {
            insert_once(&mut w, "lm_head.weight".to_string(), transpose2d(&lm))?;
        }
        // else: tied - `lm` is discarded; the embedding-derived transpose inserted above is
        // authoritative, and `insert_once` below would refuse a second write of this name anyway.
    }

    // Every block below drains (`take_dense`) the exact tensor names it owns by construction - a
    // router, a fused/split projection, a per-expert 3D tensor - before the generic per-tensor loop
    // further down lists `store`'s remaining keys. This ordering is why these blocks run first: the
    // generic loop needs no family-specific name list of its own (see the `build_weights` doc).
    if hf.is_granite_moe() {
        // MoE expert weights are stored [out, in] per expert; the moe op wants [in, out]. The router is 2D
        // ([E, hidden] -> [hidden, E]); input_linear/output_linear are 3D ([E, out, in] -> [E, in, out]).
        // None of these names end in "proj.weight", so the generic loop would otherwise take them untransposed.
        let names: Vec<String> = store.keys().map(|k| k.as_str().to_string()).collect();
        for name in &names {
            let name = name.as_str();
            if name.ends_with("router.layer.weight") {
                let rt = take_dense(store, name).with_context(|| name.to_string())?;
                insert_once(&mut w, name.to_string(), transpose2d(&rt))?;
            } else if name.ends_with("input_linear.weight")
                || name.ends_with("output_linear.weight")
            {
                let rt = take_dense(store, name).with_context(|| name.to_string())?;
                insert_once(&mut w, name.to_string(), transpose_experts(&rt))?;
            }
        }
    }
    if hf.is_qwen3_moe() {
        // Card 246: Qwen3-MoE's per-expert weights are separate safetensors tensors
        // (mlp.experts.{e}.{gate,up,down}_proj.weight), unlike granitemoe's already-fused format above (see
        // poot_models::qwen3moe's module doc). Fuse them into the two constants the tracer's `ffn()` binds, and
        // transpose the router, for every layer the decoder_sparse_step/mlp_only_layers pattern marks as routed.
        // Dense layers' mlp.{gate,up,down}_proj.weight (same names/shapes as plain qwen3 dense) are left for the
        // generic loop below.
        let n_experts = hf
            .num_experts
            .context("qwen3_moe config missing num_experts")?;
        let inter = hf
            .moe_intermediate_size
            .context("qwen3_moe config missing moe_intermediate_size")?;
        for li in 0..cfg.layers {
            if !hf.qwen3_moe_layer_is_sparse(li) {
                continue;
            }
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let router = take_dense(store, &p("mlp.gate.weight"))
                .with_context(|| format!("qwen3_moe layer {li}: missing mlp.gate.weight"))?;
            insert_once(&mut w, p("mlp.gate.weight"), transpose2d(&router))?;
            let (gate_up, down) =
                fuse_qwen3_moe_experts(store, &p("mlp.experts"), n_experts, cfg.hidden, inter)?;
            insert_once(&mut w, p("mlp.experts.gate_up_proj.weight"), gate_up)?;
            insert_once(&mut w, p("mlp.experts.down_proj.weight"), down)?;
            remove_qwen3_style_raw_experts(&mut w, &p("mlp.experts"), n_experts);
        }
    }
    if hf.is_mixtral() {
        // Card 135d: Mixtral's per-expert weights are separate safetensors tensors
        // (block_sparse_moe.experts.{e}.{w1,w2,w3}.weight), like qwen3-moe's above. Fuse them into the two
        // constants poot_models::mixtral's `mixtral_ffn` binds, and transpose the router, for every layer
        // (Mixtral has no per-layer dense/MoE switch).
        let n_experts = hf
            .num_local_experts
            .context("mixtral config missing num_local_experts")?;
        let inter = hf.intermediate_size;
        for li in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let router =
                take_dense(store, &p("block_sparse_moe.gate.weight")).with_context(|| {
                    format!("mixtral layer {li}: missing block_sparse_moe.gate.weight")
                })?;
            insert_once(
                &mut w,
                p("block_sparse_moe.gate.weight"),
                transpose2d(&router),
            )?;
            let (gate_up, down) = fuse_mixtral_experts(
                store,
                &p("block_sparse_moe.experts"),
                n_experts,
                cfg.hidden,
                inter,
            )?;
            insert_once(
                &mut w,
                p("block_sparse_moe.experts.gate_up_proj.weight"),
                gate_up,
            )?;
            insert_once(&mut w, p("block_sparse_moe.experts.down_proj.weight"), down)?;
            remove_mixtral_raw_experts(&mut w, &p("block_sparse_moe.experts"), n_experts);
        }
    }
    if hf.is_olmoe() {
        // Card 135d: OlmoE's per-expert weights are separate safetensors tensors named as qwen3-moe's
        // (mlp.experts.{e}.{gate,up,down}_proj.weight, mlp.gate.weight router; checked against the real
        // allenai/OLMoE-1B-7B-0924 model.safetensors.index.json, see poot_models::olmoe), so reuse
        // fuse_qwen3_moe_experts unchanged. Unlike qwen3_moe, every layer routes (no decoder_sparse_step/
        // mlp_only_layers), as Mixtral.
        let n_experts = hf.num_experts.context("olmoe config missing num_experts")?;
        let inter = hf.intermediate_size;
        for li in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let router = take_dense(store, &p("mlp.gate.weight"))
                .with_context(|| format!("olmoe layer {li}: missing mlp.gate.weight"))?;
            insert_once(&mut w, p("mlp.gate.weight"), transpose2d(&router))?;
            let (gate_up, down) =
                fuse_qwen3_moe_experts(store, &p("mlp.experts"), n_experts, cfg.hidden, inter)?;
            insert_once(&mut w, p("mlp.experts.gate_up_proj.weight"), gate_up)?;
            insert_once(&mut w, p("mlp.experts.down_proj.weight"), down)?;
            remove_qwen3_style_raw_experts(&mut w, &p("mlp.experts"), n_experts);
        }
    }
    if hf.is_gpt_oss() {
        // Card 135d: gpt-oss's expert weights (mlp.experts.gate_up_proj/down_proj) are already fused per layer
        // into one 3D tensor per projection, already in poot's [in,out] convention (checked against
        // tiny-random/gpt-oss's safetensors header; see poot_models::gpt_oss). The generic loop below leaves
        // them untouched (raw nn.Parameters without a ".weight" suffix), so no fuse/transpose is needed. The
        // router (`mlp.router.weight`) ends in ".weight" but not "proj.weight", so it needs the explicit
        // per-layer transpose every other arch's router gets.
        for li in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let router = take_dense(store, &p("mlp.router.weight"))
                .with_context(|| format!("gpt_oss layer {li}: missing mlp.router.weight"))?;
            insert_once(&mut w, p("mlp.router.weight"), transpose2d(&router))?;
        }
    }
    if hf.is_deepseek2() {
        // Card 135d (MLA): the routed-expert weights are separate per-expert safetensors tensors named as
        // qwen3-moe's/OlmoE's (mlp.experts.{e}.{gate,up,down}_proj.weight; checked against the
        // `yujiepan/deepseek-v2-tiny-random` safetensors header, see poot_models::deepseek2), so reuse
        // `fuse_qwen3_moe_experts` unchanged. The router (`mlp.gate.weight`) ends in ".weight" but not
        // "proj.weight" (as OlmoE's/gpt-oss's), so it needs the explicit per-layer transpose. Dense layers'
        // `mlp.{gate,up,down}_proj.weight` and the shared-expert `mlp.shared_experts.{gate,up,down}_proj.weight`
        // end in "proj.weight" and are left for the generic loop below. Attention q_a_proj/q_b_proj (or
        // q_proj), kv_b_proj, o_proj also end in "proj.weight"; q_a_layernorm/kv_a_layernorm are 1D
        // pass-through. `kv_a_proj_with_mqa.weight` is the exception: its name ends in `_with_mqa`, not
        // "proj", so `name.ends_with("proj.weight")` is false and the generic loop would otherwise leave it
        // untransposed (caught only by the real-checkpoint crosswalk test: a finite, wrong-shape matmul that
        // panics out of bounds). It needs the same explicit per-layer transpose as the router.
        for li in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let kva = take_dense(store, &p("self_attn.kv_a_proj_with_mqa.weight")).with_context(
                || format!("deepseek_v2 layer {li}: missing self_attn.kv_a_proj_with_mqa.weight"),
            )?;
            insert_once(
                &mut w,
                p("self_attn.kv_a_proj_with_mqa.weight"),
                transpose2d(&kva),
            )?;
        }
        let n_experts = hf
            .n_routed_experts
            .context("deepseek_v2 config missing n_routed_experts")?;
        let inter = hf
            .moe_intermediate_size
            .context("deepseek_v2 config missing moe_intermediate_size")?;
        let first_k_dense_replace = hf.first_k_dense_replace.unwrap_or(1);
        for li in 0..cfg.layers {
            if li < first_k_dense_replace {
                continue; // dense layer: no router/experts to fuse (generic loop below handles its MLP)
            }
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let router = take_dense(store, &p("mlp.gate.weight"))
                .with_context(|| format!("deepseek_v2 layer {li}: missing mlp.gate.weight"))?;
            insert_once(&mut w, p("mlp.gate.weight"), transpose2d(&router))?;
            let (gate_up, down) =
                fuse_qwen3_moe_experts(store, &p("mlp.experts"), n_experts, cfg.hidden, inter)?;
            insert_once(&mut w, p("mlp.experts.gate_up_proj.weight"), gate_up)?;
            insert_once(&mut w, p("mlp.experts.down_proj.weight"), down)?;
            remove_qwen3_style_raw_experts(&mut w, &p("mlp.experts"), n_experts);
        }
    }
    if hf.is_deepseek3() {
        // Mirrors the `is_deepseek2()` block above (same MLA attention, `kv_a_proj_with_mqa.weight`/
        // router-transpose/`fuse_qwen3_moe_experts` crosswalk). The one new tensor, `mlp.gate.
        // e_score_correction_bias` (a learned per-expert bias vector, not a matmul weight; see
        // `poot_models::deepseek3`), needs no special case: it does not end in "proj.weight" and is not a
        // per-expert `.experts.N.` tensor, so the generic loop below copies it through untransposed under
        // its HF name.
        for li in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let kva = take_dense(store, &p("self_attn.kv_a_proj_with_mqa.weight")).with_context(
                || format!("deepseek_v3 layer {li}: missing self_attn.kv_a_proj_with_mqa.weight"),
            )?;
            insert_once(
                &mut w,
                p("self_attn.kv_a_proj_with_mqa.weight"),
                transpose2d(&kva),
            )?;
        }
        let n_experts = hf
            .n_routed_experts
            .context("deepseek_v3 config missing n_routed_experts")?;
        let inter = hf
            .moe_intermediate_size
            .context("deepseek_v3 config missing moe_intermediate_size")?;
        let first_k_dense_replace = hf.first_k_dense_replace.unwrap_or(1);
        for li in 0..cfg.layers {
            if li < first_k_dense_replace {
                continue; // dense layer: no router/experts to fuse (generic loop below handles its MLP)
            }
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let router = take_dense(store, &p("mlp.gate.weight"))
                .with_context(|| format!("deepseek_v3 layer {li}: missing mlp.gate.weight"))?;
            insert_once(&mut w, p("mlp.gate.weight"), transpose2d(&router))?;
            let (gate_up, down) =
                fuse_qwen3_moe_experts(store, &p("mlp.experts"), n_experts, cfg.hidden, inter)?;
            insert_once(&mut w, p("mlp.experts.gate_up_proj.weight"), gate_up)?;
            insert_once(&mut w, p("mlp.experts.down_proj.weight"), down)?;
            remove_qwen3_style_raw_experts(&mut w, &p("mlp.experts"), n_experts);
        }
    }
    // Collected up front (not iterated live): `take_dense` needs `&mut store` per entry, which
    // cannot coexist with the borrow `store.keys()` would hold for the loop's duration. By this point
    // every name a block above owns has already been drained, so this loop needs no family-specific
    // skip of its own: whatever remains in `store` is exactly what the loop below should take.
    let names: Vec<String> = store.keys().map(|k| k.as_str().to_string()).collect();
    for name in &names {
        let name = name.as_str();
        if name == "model.embed_tokens.weight" {
            continue; // already handled above
        }
        // A quantized linear (GPTQ/AWQ/FP8, packed by `pack_quantized_linears` from its checkpoint
        // bytes) stays packed: its source carriers plus the WeightFormats row `bind_packed_weights`
        // reads, the tracer's `[in, out]` projection being the decode transposed (card 545a).
        if let Some(WeightEntry::Packed(payload)) = store.get(name) {
            if !name.ends_with("proj.weight") {
                bail!("{name}: a packed weight here must be a projection");
            }
            let payload = std::sync::Arc::clone(payload);
            place_packed(
                &mut w,
                &mut formats,
                name.to_string(),
                vec![payload],
                PackedLayout::Columns,
            )?;
            continue;
        }
        let rt = take_dense(store, name).with_context(|| name.to_string())?;
        let t = if name.ends_with("proj.weight") {
            transpose2d(&rt)
        } else {
            rt
        };
        insert_once(&mut w, name.to_string(), t)?;
    }
    // A LongRoPE, dynamic NTK or YaRN scaling block may omit `original_max_position_embeddings` (HF keeps it
    // at the config top level, or defaults it to `max_position_embeddings`). Fold the top-level value into
    // the scaling block the rope table builder reads, so the attention factor never divides by 0.
    let mut scaling = hf.rope_scaling.clone();
    if let Some(s) = scaling.as_mut()
        && matches!(s.rope_type.as_str(), "longrope" | "dynamic" | "yarn")
        && s.original_max_position_embeddings == 0
    {
        s.original_max_position_embeddings = hf
            .original_max_position_embeddings
            .unwrap_or(hf.max_position_embeddings);
    }
    // DeepSeek's decoupled-rope slice uses its own interleaved-pair table (half the width of the generic
    // half-split table: one angle per pair), see poot_models::deepseek2's module doc, item 6.
    // `deepseek2_yarn_params` resolves the checkpoint's YaRN scaling into the same scalars
    // `Runner::load_impl`'s `DeepseekV2Config.yarn` uses. DeepSeek-V3 reuses this table builder (its
    // MLA/YaRN shape is identical; see `is_deepseek3`), so this covers both mutually-exclusive archs.
    // Computed instead of the generic table (never both): `insert_once` refuses a second write of
    // "rope.cos"/"rope.sin", and the old "insert the generic table, then overwrite it for DeepSeek"
    // structure was the exact shape of bug this card's review flagged for `lm_head.weight` - a second
    // legitimate writer of the same name is still a second writer, so it computes the one table this
    // checkpoint needs instead of writing twice.
    let (cos, sin) = if hf.is_deepseek2() || hf.is_deepseek3() {
        let rope_dim = hf.qk_rope_head_dim.expect(
            "is_deepseek2/is_deepseek3 implies qk_rope_head_dim was validated in load_impl",
        );
        let yarn = deepseek2_yarn_params(hf);
        let (cos, sin) = poot_models::deepseek2::deepseek2_rope_tables_interleaved(
            cfg.max_pos,
            rope_dim,
            hf.effective_rope_theta()?,
            yarn.as_ref(),
        );
        let table_shape = vec![cfg.max_pos, rope_dim / 2];
        (
            HostTensor::f32(table_shape.clone(), cos),
            HostTensor::f32(table_shape, sin),
        )
    } else {
        rope_tables(cfg, hf.effective_rope_theta()?, scaling.as_ref(), None)
    };
    insert_once(&mut w, "rope.cos".to_string(), cos)?;
    insert_once(&mut w, "rope.sin".to_string(), sin)?;
    Ok((w, formats))
}

/// Fuses a Qwen3-MoE layer's separate per-expert `{prefix}.{e}.{gate,up,down}_proj.weight` safetensors
/// tensors into the two constants `poot_models::qwen3moe`'s `ffn()` helper binds: `{prefix}.gate_up_proj
/// .weight` (`[E,H,2I]`, gate||up concatenated on the last axis, transposed to `[in,out]` per expert) and
/// `{prefix}.down_proj.weight` (`[E,I,H]`, transposed to `[in,out]` per expert). The safetensors analog of
/// `poot_llm::gguf`'s `transpose_experts`/`concat_experts_last` (for granitemoe's already-3D GGUF expert
/// tensors), for tensors that arrive as `n_experts` separate 2D matrices (checked against the real
/// `yujiepin/qwen3-moe-tiny-random` checkpoint by `poot_models::qwen3moe`'s `cpu_oracle` test, which
/// prototyped this fuse).
///
/// `pub(crate)` (spec 277): `crate::deepseek32_load` reuses this for DeepSeek-V3.2's routed experts
/// (`mlp.experts.{e}.{gate,up,down}_proj.weight`, same naming as DeepSeek-V3). Within this file
/// qwen3-moe, olmoe, and deepseek2/deepseek3 share it; `deepseek32_load.rs` is the first cross-file user,
/// being a standalone module.
pub(crate) fn fuse_qwen3_moe_experts(
    store: &mut WeightStore,
    prefix: &str,
    n_experts: usize,
    hidden: usize,
    inter: usize,
) -> Result<(HostTensor, HostTensor)> {
    fuse_expert_projections(
        store,
        n_experts,
        hidden,
        inter,
        |e| format!("{prefix}.{e}.gate_proj.weight"),
        |e| format!("{prefix}.{e}.up_proj.weight"),
        |e| format!("{prefix}.{e}.down_proj.weight"),
    )
}

/// Fuses a Mixtral layer's separate per-expert `{prefix}.{e}.{w1,w2,w3}.weight` safetensors tensors into
/// the two constants `poot_models::mixtral`'s `mixtral_ffn` binds: `{prefix}.gate_up_proj.weight`
/// (`[E,H,2I]`, gate||up concatenated on the last axis, transposed to `[in,out]` per expert) and
/// `{prefix}.down_proj.weight` (`[E,I,H]`, transposed to `[in,out]` per expert). `w1` = gate
/// (SiLU-activated), `w3` = up, `w2` = down, per `MixtralBlockSparseTop2MLP.forward`:
/// `w2(act_fn(w1(x)) * w3(x))` (spec 261). Identical to `fuse_qwen3_moe_experts` modulo the name mapping
/// (Mixtral uses the older Mistral-family per-expert names).
pub(crate) fn fuse_mixtral_experts(
    store: &mut WeightStore,
    prefix: &str,
    n_experts: usize,
    hidden: usize,
    inter: usize,
) -> Result<(HostTensor, HostTensor)> {
    fuse_expert_projections(
        store,
        n_experts,
        hidden,
        inter,
        |e| format!("{prefix}.{e}.w1.weight"),
        |e| format!("{prefix}.{e}.w3.weight"),
        |e| format!("{prefix}.{e}.w2.weight"),
    )
}

/// The shared body of [`fuse_qwen3_moe_experts`]/[`fuse_mixtral_experts`]: drains each expert's
/// gate/up/down tensors (HF `[out, in]`) by the given names and fuses them, in their stored
/// dtype, into `gate_up` (`[E,H,2I]`, gate||up on the last axis) and `down` (`[E,I,H]`), both
/// transposed to `[in,out]` per expert.
fn fuse_expert_projections(
    store: &mut WeightStore,
    n_experts: usize,
    hidden: usize,
    inter: usize,
    gate_name: impl Fn(usize) -> String,
    up_name: impl Fn(usize) -> String,
    down_name: impl Fn(usize) -> String,
) -> Result<(HostTensor, HostTensor)> {
    let mut take = |name: String| take_dense(store, &name).with_context(|| name.clone());
    let mut gates_ups = Vec::with_capacity(2 * n_experts);
    let mut downs = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        gates_ups.push(take(gate_name(e))?);
    }
    for e in 0..n_experts {
        gates_ups.push(take(up_name(e))?);
    }
    for e in 0..n_experts {
        downs.push(take(down_name(e))?);
    }
    let n = n_experts;
    // Sources `0..n` are the gates and `n..2n` the ups, each `[inter, hidden]`: output element
    // `[e, row, col]` is the transposed gate (`col < inter`) or up (`col >= inter`).
    let gate_up_sources: Vec<&HostTensor> = gates_ups.iter().collect();
    let gate_up = gather_from(
        &gate_up_sources,
        vec![n, hidden, 2 * inter],
        (0..n).flat_map(move |e| {
            (0..hidden).flat_map(move |row| {
                (0..2 * inter).map(move |col| {
                    if col < inter {
                        (e, col * hidden + row)
                    } else {
                        (n + e, (col - inter) * hidden + row)
                    }
                })
            })
        }),
    );
    // Each down is `[hidden, inter]`; the output `[e, i, h]` is its transpose.
    let down_sources: Vec<&HostTensor> = downs.iter().collect();
    let down = gather_from(
        &down_sources,
        vec![n, inter, hidden],
        (0..n).flat_map(move |e| {
            (0..inter).flat_map(move |i| (0..hidden).map(move |h| (e, h * inter + i)))
        }),
    );
    Ok((gate_up, down))
}

/// Defensive no-op cleanup for the raw per-expert `{prefix}.{e}.{gate,up,down}_proj.weight` entries:
/// call right after [`fuse_qwen3_moe_experts`] has folded the same tensors into the fused
/// `{prefix}.gate_up_proj.weight`/`{prefix}.down_proj.weight` constants the tracer binds.
///
/// Originally (card 264 update 0633) the generic loop inserted every raw per-expert tensor into `w`
/// (their names end in "proj.weight"), leaving both raw and fused copies resident: dead memory scaling
/// with `n_experts * moe_layers` (~86GB for `deepseek-ai/DeepSeek-V2-Lite-Chat`'s 64-expert/26-MoE-layer
/// checkpoint: 14.4B routed-expert params x 6 bytes), which OOM-killed a RunPod pod with a ~234GB
/// container ceiling. That fixed only the steady-state leak; peak memory during `build_weights` was
/// unchanged because the generic loop still transposed and bf16-cloned every raw per-expert tensor
/// before this function ran. The generic loop now skips them ([`is_raw_moe_expert_tensor`]), so the
/// `w.remove()` calls are normally no-ops; kept as a defensive second layer (harmless on a missing key)
/// in case a future checkpoint's naming slips past the skip check.
pub(crate) fn remove_qwen3_style_raw_experts(
    w: &mut HashMap<String, Value>,
    prefix: &str,
    n_experts: usize,
) {
    for e in 0..n_experts {
        w.remove(&format!("{prefix}.{e}.gate_proj.weight"));
        w.remove(&format!("{prefix}.{e}.up_proj.weight"));
        w.remove(&format!("{prefix}.{e}.down_proj.weight"));
    }
}

/// The [`remove_qwen3_style_raw_experts`] cleanup for Mixtral's per-expert names (`w1`/`w2`/`w3`; see
/// [`fuse_mixtral_experts`]). Same rationale, and likewise a defensive no-op now that the generic loop
/// skips these names.
pub(crate) fn remove_mixtral_raw_experts(
    w: &mut HashMap<String, Value>,
    prefix: &str,
    n_experts: usize,
) {
    for e in 0..n_experts {
        w.remove(&format!("{prefix}.{e}.w1.weight"));
        w.remove(&format!("{prefix}.{e}.w2.weight"));
        w.remove(&format!("{prefix}.{e}.w3.weight"));
    }
}

/// The baked [`WeightStore`] `Runner::load_on` (Card 546a's one Runner consumer) gives
/// `poot_executor::Executor::load_weights`: every `weights` value re-expressed as a
/// [`WeightEntry`], keyed exactly as the decode/prefill graphs' `Storage::Const` names already are
/// (`Value::Host` by its own name; a `Value::Packed` by its [`PackedSourceName`]'s
/// `linear_id`, Card 642's const naming -), so the engine's generic (store key, planned
/// storage) residency binds every const with no Runner-specific knowledge. Every role-component of
/// one packed linear shares its owner's identical `Arc<PackedPayload>` ([`place_packed`] clones it
/// per role), so the first role seen for a `linear_id` inserts it and later roles are a no-op, not a
/// duplicate-key error.
pub(crate) fn weight_store(weights: &HashMap<String, Value>) -> Result<WeightStore> {
    let mut builder = WeightStore::builder();
    let mut packed_linear_ids: HashSet<String> = HashSet::new();
    for (name, value) in weights {
        match value {
            Value::Host(t) => {
                let dense = DenseWeight::try_new(
                    t.dtype(),
                    t.shape().to_vec(),
                    Arc::from(t.view().bytes()),
                )
                .with_context(|| format!("{name}: baking the executor's weight store"))?;
                builder
                    .insert(name.clone(), WeightEntry::Dense(dense))
                    .with_context(|| {
                        format!("{name}: duplicate dense weight in the baked store")
                    })?;
            }
            Value::Packed(component) => {
                let parsed = PackedSourceName::parse(name).with_context(|| {
                    format!("{name}: packed weight value whose name is not a packed source name")
                })?;
                if packed_linear_ids.insert(parsed.linear_id().to_string()) {
                    builder
                        .insert(
                            parsed.linear_id().to_string(),
                            WeightEntry::Packed(Arc::clone(component.owner())),
                        )
                        .with_context(|| {
                            format!(
                                "{}: duplicate packed weight in the baked store",
                                parsed.linear_id()
                            )
                        })?;
                }
            }
            Value::Owner(_) => {
                bail!(
                    "{name}: this weight value kind has no poot-executor WeightStore encoding yet"
                )
            }
        }
    }
    Ok(builder.build())
}

/// Why a traced graph cannot take the dtype its checkpoint stores for a dense const (Card 1008).
#[derive(Debug, thiserror::Error)]
pub(crate) enum StoredDtypeError {
    /// A stored dtype the graph cannot read as declared: only a stored BF16/F16 const declared F32 has a
    /// reading (an explicit in-graph cast, or native consumption by a contraction).
    #[error("{name}: stored as {stored} but the graph declares {declared}")]
    Mismatch {
        name: String,
        stored: DType,
        declared: DType,
    },
    /// Carrying the stored dtype of `consts` through their views, or reading it through a cast, left the
    /// graph ill-typed.
    #[error("retyping {consts:?} to their stored dtypes: {source}")]
    Graph {
        consts: Vec<String>,
        source: Box<poot_graph_ir::GraphValidationError>,
    },
}

/// Ops that only move a weight's elements, so a const reaching a contraction through them is a contraction
/// weight and keeps its dtype through them. A gather is not one: its rows are activations (a token embedding),
/// and a gather's table is read through the cast-after-gather instead.
fn passes_weight_through(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::Reshape { .. }
            | OpKind::Transpose { .. }
            | OpKind::Slice { .. }
            | OpKind::Concat { .. }
            | OpKind::Broadcast { .. }
    )
}

fn is_contraction(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::MatMul
            | OpKind::MatMulBias
            | OpKind::IndexedMatMul
            | OpKind::DenseContraction { .. }
            | OpKind::DenseRowGather { .. }
    )
}

/// The values `root` reaches through [`passes_weight_through`] ops (root excluded) when some contraction
/// reads `root` or one of them; `None` when no contraction does.
fn contraction_views(g: &Graph, root: ValueId) -> Option<Vec<ValueId>> {
    let mut views = vec![root];
    let mut feeds = false;
    let mut next = 0;
    while next < views.len() {
        let value = views[next];
        next += 1;
        for eqn in &g.eqns {
            if !eqn.inputs.iter().any(|operand| is_value(operand, value)) {
                continue;
            }
            if is_contraction(&eqn.op) {
                feeds = true;
            } else if passes_weight_through(&eqn.op) && !views.contains(&eqn.out) {
                views.push(eqn.out);
            }
        }
    }
    feeds.then(|| views.split_off(1))
}

/// Put the dtypes `weights` stores onto the dense consts of a traced graph (Card 1008). A tracer declares a
/// norm scale, an embedding table, a bias or a head at its logical dtype F32, while a BF16/F16 checkpoint
/// stores 16-bit words. Each such const (declared F32, stored BF16 or F16, not packed) is retyped to its
/// stored dtype, so the executors upload the stored words and nothing widens on upload or at bind:
///
/// - a const only elementwise code reads is read through an explicit `Cast` to F32, a planned equation (a
///   gather's table stays stored and the cast moves after the gather, so only the gathered rows widen);
/// - a const a contraction reads (through views or a concat) gets NO cast: it takes its stored dtype
///   through those views and the contraction consumes it as stored, the way a `proj_dtype` tracer declares it.
///
/// A const that matches its stored dtype is untouched; any other disagreement is
/// [`StoredDtypeError::Mismatch`].
pub(crate) fn bind_stored_dtypes(
    g: &Graph,
    weights: &HashMap<String, Value>,
    formats: &WeightFormats,
) -> std::result::Result<Graph, StoredDtypeError> {
    let mut out = g.clone();
    let mut retyped: Vec<String> = Vec::new();
    let mut cast_consts: Vec<ValueId> = Vec::new();
    for &id in &g.consts {
        let meta = g.meta(id);
        let Some(name) = meta.name.as_deref() else {
            continue;
        };
        if formats.get(name).is_some() {
            continue;
        }
        let Some(Value::Host(stored)) = weights.get(name) else {
            continue;
        };
        let (stored, declared) = (stored.dtype(), meta.aval.dtype);
        if stored == declared {
            continue;
        }
        if !(matches!(stored, DType::BF16 | DType::F16) && declared == DType::F32) {
            return Err(StoredDtypeError::Mismatch {
                name: name.to_string(),
                stored,
                declared,
            });
        }
        out.values[id].aval.dtype = stored;
        retyped.push(name.to_string());
        match contraction_views(g, id) {
            Some(views) => {
                for view in views {
                    out.values[view].aval.dtype = stored;
                }
            }
            None => cast_consts.push(id),
        }
    }
    if retyped.is_empty() {
        return Ok(out);
    }
    if cast_consts.is_empty() {
        out.validate().map_err(|source| StoredDtypeError::Graph {
            consts: retyped,
            source: Box::new(source),
        })?;
        return Ok(out);
    }
    let mut casts: Vec<Eqn> = Vec::new();
    let mut whole: HashMap<ValueId, ValueId> = HashMap::new();
    for &id in &cast_consts {
        let used_whole = g.eqns.iter().any(|eqn| {
            eqn.inputs
                .iter()
                .enumerate()
                .any(|(slot, operand)| is_value(operand, id) && !is_gather_table(eqn, slot))
        });
        if used_whole {
            let shape = g.meta(id).aval.shape.clone();
            out.values.push(ValueMeta::new(
                TensorType::new(shape, DType::F32),
                Storage::Device,
                None,
            ));
            let cast = out.values.len() - 1;
            casts.push(Eqn {
                op: OpKind::Cast { to: DType::F32 },
                inputs: vec![Operand::Value(id)],
                out: cast,
                layer: None,
            });
            whole.insert(id, cast);
        }
    }
    let mut eqns = Vec::with_capacity(g.eqns.len() + casts.len());
    eqns.append(&mut casts);
    for eqn in &g.eqns {
        let mut eqn = eqn.clone();
        let table = match (&eqn.op, eqn.inputs.first()) {
            (OpKind::Gather { .. }, Some(&Operand::Value(table)))
                if cast_consts.contains(&table) =>
            {
                Some(table)
            }
            _ => None,
        };
        let gather = matches!(eqn.op, OpKind::Gather { .. });
        for (slot, operand) in eqn.inputs.iter_mut().enumerate() {
            if let Operand::Value(value) = operand
                && let Some(&cast) = whole.get(value)
                && !(gather && slot == 0)
            {
                *operand = Operand::Value(cast);
            }
        }
        match table {
            Some(table) => {
                let stored = out.values[table].aval.dtype;
                let rows = out.values[eqn.out].aval.shape.clone();
                out.values.push(ValueMeta::new(
                    TensorType::new(rows, stored),
                    Storage::Device,
                    None,
                ));
                let gathered = out.values.len() - 1;
                let (original, layer) = (eqn.out, eqn.layer);
                eqn.out = gathered;
                eqns.push(eqn);
                eqns.push(Eqn {
                    op: OpKind::Cast { to: DType::F32 },
                    inputs: vec![Operand::Value(gathered)],
                    out: original,
                    layer,
                });
            }
            None => eqns.push(eqn),
        }
    }
    out.eqns = eqns;
    out.validate().map_err(|source| StoredDtypeError::Graph {
        consts: retyped,
        source: Box::new(source),
    })?;
    Ok(out)
}

fn is_value(operand: &Operand, value: ValueId) -> bool {
    matches!(operand, Operand::Value(v) if *v == value)
}

fn is_gather_table(eqn: &Eqn, slot: usize) -> bool {
    slot == 0 && matches!(eqn.op, OpKind::Gather { .. })
}
