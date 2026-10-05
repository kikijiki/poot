//! candle baseline runner for the poot benchmark suite (suite contract, spec 112).
//!
//! candle's per-arch APIs differ slightly (Qwen2/Qwen3/Mistral share `ModelForCausalLM::new` +
//! `forward(&x, offset)`; Llama uses an explicit `Cache`), so the dispatch covers the dense arches in the
//! manifest. Generates exactly `--gen-tokens` greedily (ignores EOS) for comparable counts, times
//! TTFT/decode (sample() reads logits to host, which syncs), and prints one JSON line as the last stdout
//! line. VRAM is measured externally by the harness observer. See NOTES.md for the precision/arch support
//! matrix (no GPTQ/AWQ, no SmolVLM/OLMoE/gpt-oss in 0.10.2).

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::time::Instant;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// The dense arches this runner dispatches. Each shares the `new(&cfg, vb)` + `forward(&x, offset)` shape
/// except Llama (its own `Cache`), handled in `run_model`.
enum Arch {
    Qwen2,
    Qwen3,
    Mistral,
    Llama,
}

fn detect_arch(dir: &str) -> Result<Arch> {
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/config.json"))?)?;
    let arches = cfg["architectures"].as_array().and_then(|a| a.first()).and_then(|s| s.as_str())
        .unwrap_or("");
    Ok(match arches {
        "Qwen2ForCausalLM" => Arch::Qwen2,
        "Qwen3ForCausalLM" => Arch::Qwen3,
        "MistralForCausalLM" => Arch::Mistral,
        "LlamaForCausalLM" => Arch::Llama,
        other => return Err(anyhow!("candle runner: unsupported arch {other:?}")),
    })
}

fn weight_files(dir: &str) -> Result<Vec<std::path::PathBuf>> {
    let idx = std::path::Path::new(dir).join("model.safetensors.index.json");
    if idx.exists() {
        let j: serde_json::Value = serde_json::from_slice(&std::fs::read(&idx)?)?;
        let mut files = std::collections::BTreeSet::new();
        for v in j["weight_map"].as_object().unwrap().values() {
            files.insert(v.as_str().unwrap().to_string());
        }
        Ok(files.into_iter().map(|f| std::path::Path::new(dir).join(f)).collect())
    } else {
        Ok(vec![std::path::Path::new(dir).join("model.safetensors")])
    }
}

fn dtype_of(precision: &str) -> DType {
    match precision {
        "f16" | "fp16" => DType::F16,
        "f32" | "fp32" => DType::F32,
        _ => DType::BF16,
    }
}

/// Run prefill + exactly `gen_tokens` decode steps. Returns (ttft_ms, tpot_ms, e2e_ms, decode_tok_s).
/// `forward` is `FnMut(&[u32] tokens, usize offset) -> Result<Tensor [vocab]>`.
fn timed_generate(
    device: &Device,
    prompt_ids: &[u32],
    gen_tokens: usize,
    mut forward: impl FnMut(&[u32], usize) -> Result<Tensor>,
) -> Result<(f64, f64, f64, f64)> {
    let mut lp = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let mut tokens = prompt_ids.to_vec();
    device.synchronize()?;
    let t0 = Instant::now();
    let mut t_first = None;
    let mut t_prev = t0;
    let mut tpots = Vec::new();
    for i in 0..gen_tokens {
        let (ctxt, offset) = if i == 0 {
            (&tokens[..], 0)
        } else {
            (&tokens[tokens.len() - 1..], tokens.len() - 1)
        };
        let logits = forward(ctxt, offset)?;
        let next = lp.sample(&logits)?; // reads logits to host -> a device sync, so the Instant below is accurate
        let now = Instant::now();
        if i == 0 {
            t_first = Some(now);
        } else {
            tpots.push((now - t_prev).as_secs_f64() * 1000.0);
        }
        t_prev = now;
        tokens.push(next);
    }
    device.synchronize()?;
    let t_end = Instant::now();
    let first = t_first.unwrap();
    let ttft = (first - t0).as_secs_f64() * 1000.0;
    let e2e = (t_end - t0).as_secs_f64() * 1000.0;
    let tpot = if tpots.is_empty() { 0.0 } else { tpots.iter().sum::<f64>() / tpots.len() as f64 };
    let decode = gen_tokens as f64 / (e2e / 1000.0);
    Ok((ttft, tpot, e2e, decode))
}

/// decode-curve (spec 115): prefill `prompt_ids`, decode `osl` tokens, time each. Returns raw
/// (ttft_ms, e2e_ms, per-token itl_ms). `sample()` syncs by reading logits to host.
fn timed_curve(
    device: &Device,
    prompt_ids: &[u32],
    osl: usize,
    mut forward: impl FnMut(&[u32], usize) -> Result<Tensor>,
) -> Result<(f64, f64, Vec<f64>)> {
    let mut lp = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let mut tokens = prompt_ids.to_vec();
    device.synchronize()?;
    let t0 = Instant::now();
    let logits = forward(&tokens[..], 0)?;
    let next = lp.sample(&logits)?;
    let first = Instant::now();
    let ttft = (first - t0).as_secs_f64() * 1000.0;
    tokens.push(next);
    let mut itls = Vec::with_capacity(osl.saturating_sub(1));
    let mut t_prev = first;
    for _ in 0..osl.saturating_sub(1) {
        let off = tokens.len() - 1;
        let logits = forward(&tokens[off..], off)?;
        let next = lp.sample(&logits)?;
        let now = Instant::now();
        itls.push((now - t_prev).as_secs_f64() * 1000.0);
        t_prev = now;
        tokens.push(next);
    }
    device.synchronize()?;
    let e2e = (Instant::now() - t0).as_secs_f64() * 1000.0;
    Ok((ttft, e2e, itls))
}

/// Like `run_model` for one decode-curve decode (prefill ids, decode `osl`, raw per-token timing). Reloads
/// the model (mmap) so each call starts with a clean KV cache, as `run_model` does.
fn run_model_curve(
    arch: &Arch,
    dir: &str,
    device: &Device,
    dtype: DType,
    prompt_ids: &[u32],
    osl: usize,
) -> Result<(f64, f64, Vec<f64>)> {
    let files = weight_files(dir)?;
    let cfg_bytes = std::fs::read(format!("{dir}/config.json"))?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
    let dev = device.clone();
    let mk = |t: &[u32]| -> Result<Tensor> { Ok(Tensor::new(t, &dev)?.unsqueeze(0)?) };
    match arch {
        Arch::Qwen2 => {
            use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
            let cfg: Config = serde_json::from_slice(&cfg_bytes)?;
            let mut m = ModelForCausalLM::new(&cfg, vb)?;
            timed_curve(device, prompt_ids, osl, |t, off| {
                Ok(m.forward(&mk(t)?, off)?.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?)
            })
        }
        Arch::Qwen3 => {
            use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
            let cfg: Config = serde_json::from_slice(&cfg_bytes)?;
            let mut m = ModelForCausalLM::new(&cfg, vb)?;
            timed_curve(device, prompt_ids, osl, |t, off| {
                Ok(m.forward(&mk(t)?, off)?.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?)
            })
        }
        Arch::Mistral => {
            use candle_transformers::models::mistral::{Config, Model};
            let cfg: Config = serde_json::from_slice(&cfg_bytes)?;
            let mut m = Model::new(&cfg, vb)?;
            timed_curve(device, prompt_ids, osl, |t, off| {
                Ok(m.forward(&mk(t)?, off)?.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?)
            })
        }
        Arch::Llama => {
            use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
            let lcfg: LlamaConfig = serde_json::from_slice(&cfg_bytes)?;
            let cfg = lcfg.into_config(false);
            let mut cache = Cache::new(true, dtype, &cfg, device)?;
            let m = Llama::load(vb, &cfg)?;
            timed_curve(device, prompt_ids, osl, |t, off| {
                Ok(m.forward(&mk(t)?, off, &mut cache)?.squeeze(0)?.to_dtype(DType::F32)?)
            })
        }
    }
}

/// Deterministic in-vocab token ids (LCG) for synthetic decode-curve prefill (tokenizer-neutral).
fn synth_ids(n: usize, vocab: u32) -> Vec<u32> {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as u32) % vocab.max(1)
        })
        .collect()
}

fn run_model(
    arch: Arch,
    dir: &str,
    device: &Device,
    dtype: DType,
    prompt_ids: &[u32],
    gen_tokens: usize,
) -> Result<(f64, f64, f64, f64)> {
    let files = weight_files(dir)?;
    let cfg_bytes = std::fs::read(format!("{dir}/config.json"))?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
    let dev = device.clone();
    // Build the [1, seqlen] input tensor for a step (offset is passed to forward separately).
    let mk = |t: &[u32]| -> Result<Tensor> { Ok(Tensor::new(t, &dev)?.unsqueeze(0)?) };
    match arch {
        Arch::Qwen2 => {
            use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
            let cfg: Config = serde_json::from_slice(&cfg_bytes)?;
            let mut m = ModelForCausalLM::new(&cfg, vb)?;
            timed_generate(device, prompt_ids, gen_tokens, |t, off| {
                let logits = m.forward(&mk(t)?, off)?;
                Ok(logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?)
            })
        }
        Arch::Qwen3 => {
            use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
            let cfg: Config = serde_json::from_slice(&cfg_bytes)?;
            let mut m = ModelForCausalLM::new(&cfg, vb)?;
            timed_generate(device, prompt_ids, gen_tokens, |t, off| {
                let logits = m.forward(&mk(t)?, off)?;
                Ok(logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?)
            })
        }
        Arch::Mistral => {
            use candle_transformers::models::mistral::{Config, Model};
            let cfg: Config = serde_json::from_slice(&cfg_bytes)?;
            let mut m = Model::new(&cfg, vb)?;
            timed_generate(device, prompt_ids, gen_tokens, |t, off| {
                let logits = m.forward(&mk(t)?, off)?;
                Ok(logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?)
            })
        }
        Arch::Llama => {
            use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
            let lcfg: LlamaConfig = serde_json::from_slice(&cfg_bytes)?;
            let cfg = lcfg.into_config(false); // use_flash_attn = false
            let mut cache = Cache::new(true, dtype, &cfg, device)?;
            let m = Llama::load(vb, &cfg)?;
            timed_generate(device, prompt_ids, gen_tokens, |t, off| {
                let logits = m.forward(&mk(t)?, off, &mut cache)?;
                Ok(logits.squeeze(0)?.to_dtype(DType::F32)?)
            })
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = flag(&args, "--model-dir").ok_or_else(|| anyhow!("--model-dir required"))?;
    let prompt_file = flag(&args, "--prompt-file").ok_or_else(|| anyhow!("--prompt-file required"))?;
    let gen_tokens: usize = flag(&args, "--gen-tokens").and_then(|s| s.parse().ok()).unwrap_or(128);
    let warmup: usize = flag(&args, "--warmup").and_then(|s| s.parse().ok()).unwrap_or(2);
    let iters: usize = flag(&args, "--iters").and_then(|s| s.parse().ok()).unwrap_or(5);
    let precision = flag(&args, "--precision").unwrap_or_else(|| "bf16".to_string());

    let device = Device::new_cuda(0)?;
    let dtype = dtype_of(&precision);

    // decode-curve mode (spec 115): sweep ISL, decode OSL per point, emit raw per-token timing as curve JSON.
    let mode = flag(&args, "--mode").unwrap_or_else(|| "single".to_string());
    if mode == "decode-curve" {
        let osl: usize = flag(&args, "--osl").and_then(|s| s.parse().ok()).unwrap_or(128);
        let isls: Vec<usize> = flag(&args, "--isl-list")
            .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![128usize, 512, 2048, 8192]);
        let synthetic = args.iter().any(|a| a == "--synthetic");
        let arch = detect_arch(&dir)?;
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(format!("{dir}/config.json"))?)?;
        let vocab = cfg["vocab_size"].as_u64().unwrap_or(32000) as u32;
        eprintln!("candle decode-curve: osl={osl} isls={isls:?} synthetic={synthetic} vocab={vocab}");
        let mut curve = Vec::new();
        for &isl in &isls {
            let ids: Vec<u32> = if synthetic {
                synth_ids(isl, vocab)
            } else {
                let tk = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json"))
                    .map_err(|e| anyhow!("tokenizer: {e}"))?;
                let text = std::fs::read_to_string(&prompt_file).unwrap_or_default();
                let base = tk.encode(text, true).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec();
                let blen = base.len().max(1);
                (0..isl).map(|i| base[i % blen]).collect()
            };
            for _ in 0..warmup {
                run_model_curve(&arch, &dir, &device, dtype, &ids, osl)?;
            }
            let mut samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                let (ttft, e2e, itls) = run_model_curve(&arch, &dir, &device, dtype, &ids, osl)?;
                samples.push(serde_json::json!({"ttft_ms": ttft, "e2e_ms": e2e, "itl_ms": itls}));
            }
            eprintln!("candle decode-curve: isl={isl} done ({iters} iters)");
            curve.push(serde_json::json!({"isl": isl, "prompt_tokens": ids.len(), "iters": samples}));
        }
        let out = serde_json::json!({
            "framework": "candle", "framework_version": "0.10.2", "precision": precision,
            "mode": "decode-curve", "osl": osl, "curve": curve,
        });
        println!("{}", serde_json::to_string(&out)?);
        return Ok(());
    }

    let prompt = std::fs::read_to_string(&prompt_file)?.trim_end().to_string();

    let tk = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json"))
        .map_err(|e| anyhow!("tokenizer: {e}"))?;
    let prompt_ids: Vec<u32> = tk.encode(prompt, true).map_err(|e| anyhow!("encode: {e}"))?
        .get_ids().to_vec();

    for w in 0..warmup {
        let arch = detect_arch(&dir)?;
        run_model(arch, &dir, &device, dtype, &prompt_ids, gen_tokens)?;
        eprintln!("candle: warmup {}/{warmup}", w + 1);
    }

    let (mut ttfts, mut tpots, mut e2es, mut decs) = (vec![], vec![], vec![], vec![]);
    for it in 0..iters {
        let arch = detect_arch(&dir)?;
        let (ttft, tpot, e2e, decode) = run_model(arch, &dir, &device, dtype, &prompt_ids, gen_tokens)?;
        ttfts.push(ttft);
        tpots.push(tpot);
        e2es.push(e2e);
        decs.push(decode);
        eprintln!("candle: iter {}/{iters}  TTFT {ttft:.0}ms  {decode:.1} tok/s", it + 1);
    }

    let median = |xs: &[f64]| {
        let mut v = xs.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let stdev = |xs: &[f64]| {
        if xs.len() < 2 {
            return 0.0;
        }
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
    };
    println!(
        "{{\"framework\":\"candle\",\"framework_version\":\"0.10.2\",\"precision\":\"{}\",\
\"prompt_tokens\":{},\"gen_tokens\":{},\"iterations\":{},\"ttft_ms\":{:.3},\"tpot_ms\":{:.3},\
\"e2e_ms\":{:.3},\"decode_tok_s\":{:.3},\"ttft_ms_stdev\":{:.3},\"decode_tok_s_stdev\":{:.3}}}",
        precision, prompt_ids.len(), gen_tokens, iters,
        median(&ttfts), median(&tpots), median(&e2es), median(&decs), stdev(&ttfts), stdev(&decs),
    );
    Ok(())
}
