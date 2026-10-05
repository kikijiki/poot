use super::*;

#[test]
fn vision_encoder_smoke_and_executor_equiv() {
    // `trace_vision_encoder` (patch embed, N blocks, final LayerNorm) on a tiny 2-layer config: output shape [1, num_patches, hidden], finite, GPU == CPU.
    use poot_models::vision::{
        MhaWeights, VisionBlockWeights, VisionConfig, VisionEncoderWeights, trace_vision_encoder,
    };
    let cfg = VisionConfig {
        channels: 2,
        image_size: 4,
        patch_size: 2,
        n_heads: 2,
        head_dim: 2,
        intermediate: 6,
        layers: 2,
        eps: 1e-5,
    };
    let h = cfg.hidden(); // 4
    let np = cfg.num_patches(); // 4
    let pd = cfg.channels * cfg.patch_size * cfg.patch_size; // 8
    let inter = 6usize;
    let b = Builder::new();
    let mut binds = Vec::new();
    let pixels = vconst(
        &b,
        &mut binds,
        "pix",
        vec![cfg.channels, cfg.image_size, cfg.image_size],
    );
    let blk = |i: usize, binds: &mut Vec<_>| VisionBlockWeights {
        ln1_w: vconst(&b, binds, &format!("l1w{i}"), vec![h]),
        ln1_b: vconst(&b, binds, &format!("l1b{i}"), vec![h]),
        mha: MhaWeights {
            wq: vconst(&b, binds, &format!("wq{i}"), vec![h, h]),
            bq: vconst(&b, binds, &format!("bq{i}"), vec![h]),
            wk: vconst(&b, binds, &format!("wk{i}"), vec![h, h]),
            bk: vconst(&b, binds, &format!("bk{i}"), vec![h]),
            wv: vconst(&b, binds, &format!("wv{i}"), vec![h, h]),
            bv: vconst(&b, binds, &format!("bv{i}"), vec![h]),
            wo: vconst(&b, binds, &format!("wo{i}"), vec![h, h]),
            bo: vconst(&b, binds, &format!("bo{i}"), vec![h]),
        },
        ln2_w: vconst(&b, binds, &format!("l2w{i}"), vec![h]),
        ln2_b: vconst(&b, binds, &format!("l2b{i}"), vec![h]),
        fc1_w: vconst(&b, binds, &format!("f1w{i}"), vec![h, inter]),
        fc1_b: vconst(&b, binds, &format!("f1b{i}"), vec![inter]),
        fc2_w: vconst(&b, binds, &format!("f2w{i}"), vec![inter, h]),
        fc2_b: vconst(&b, binds, &format!("f2b{i}"), vec![h]),
    };
    let w = VisionEncoderWeights {
        patch_w: vconst(&b, &mut binds, "pw", vec![pd, h]),
        patch_b: vconst(&b, &mut binds, "pb", vec![h]),
        pos_embed: vconst(&b, &mut binds, "pos", vec![np, h]),
        blocks: (0..cfg.layers).map(|i| blk(i, &mut binds)).collect(),
        post_ln_w: vconst(&b, &mut binds, "plw", vec![h]),
        post_ln_b: vconst(&b, &mut binds, "plb", vec![h]),
    };
    let out = trace_vision_encoder(&b, pixels, &w, &cfg);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    for (k, (id, shape)) in binds.iter().enumerate() {
        let n: usize = shape.iter().product();
        // small values keep the deep composition numerically tame; LN weights near 1.
        let d: Vec<f32> = (0..n)
            .map(|j| ((j + k * 7 + 1) as f32 * 0.3).sin() * 0.4)
            .collect();
        inputs.insert(*id, HostTensor::f32(shape.clone(), d));
    }

    let eval_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(cpu.shape(), vec![1, np, h], "encoder output shape");
    assert!(
        cpu.as_f32().unwrap().iter().all(|v| v.is_finite()),
        "encoder output not finite"
    );
    let _gpu_guard = gpu_lock();
    if let Some((mut gpu, target)) = open_engine_or_skip() {
        let got = run_via_contract(&mut gpu, target, &g, &inputs);
        assert_eq!(got.shape(), cpu.shape());
        for (i, (a, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap())
            .enumerate()
        {
            assert!((a - c).abs() <= 1e-4, "encoder gpu elem {i}: {a} vs {c}");
        }
    }
}

#[test]
fn bert_embeddings_match_reference_and_encoder_lowers() {
    // BERT embeddings = word[tokens] + position[0..L] + token_type[0], then LayerNorm, match a hand reference; the full encoder
    // (embeddings + blocks) lowers GPU == CPU.
    use poot_models::encoder::{
        BertConfig, BertEncoderWeights, trace_bert_embeddings, trace_bert_encoder,
    };
    use poot_models::vision::{MhaWeights, VisionBlockWeights};
    let (vocab, h, seq) = (10usize, 4usize, 3usize);
    let eps = 1e-12f32;
    // ---- embeddings vs reference ----
    let b = Builder::new();
    let tokens = b.constant("tok", TensorType::f32(vec![seq]));
    let we = b.constant("we", TensorType::f32(vec![vocab, h]));
    let pe = b.constant("pe", TensorType::f32(vec![8, h]));
    let te = b.constant("te", TensorType::f32(vec![2, h]));
    let lw = b.constant("lw", TensorType::f32(vec![h]));
    let lb = b.constant("lb", TensorType::f32(vec![h]));
    let emb = trace_bert_embeddings(&b, tokens, we, pe, te, lw, lb, seq, h, eps);
    let g = b.finish(emb);

    let r =
        |s: usize, n: usize| -> Vec<f32> { (0..n).map(|i| ((i + s) as f32 * 0.3).sin()).collect() };
    let toks = [1u32, 5, 2];
    let wed = r(1, vocab * h);
    let ped = r(2, 8 * h);
    let ted = r(3, 2 * h);
    let lwd = vec![1.0f32; h];
    let lbd = r(4, h);
    let mut inp = HashMap::new();
    inp.insert(
        tokens.id,
        Value::from(HostTensor::f32(
            vec![seq],
            toks.iter().map(|&t| t as f32).collect(),
        )),
    );
    inp.insert(
        we.id,
        Value::from(HostTensor::f32(vec![vocab, h], wed.clone())),
    );
    inp.insert(pe.id, Value::from(HostTensor::f32(vec![8, h], ped.clone())));
    inp.insert(te.id, Value::from(HostTensor::f32(vec![2, h], ted.clone())));
    inp.insert(lw.id, Value::from(HostTensor::f32(vec![h], lwd.clone())));
    inp.insert(lb.id, Value::from(HostTensor::f32(vec![h], lbd.clone())));
    let got = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(got.shape(), vec![1, seq, h]);
    // reference: sum then LayerNorm per row.
    for (i, &t) in toks.iter().enumerate() {
        let mut row = vec![0.0f32; h];
        for j in 0..h {
            row[j] = wed[t as usize * h + j] + ped[i * h + j] + ted[j]; // token type 0
        }
        let mean = row.iter().sum::<f32>() / h as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / h as f32;
        let den = (var + eps).sqrt();
        for j in 0..h {
            let want = (row[j] - mean) / den * lwd[j] + lbd[j];
            assert!(
                (got.as_f32().unwrap()[i * h + j] - want).abs() < 1e-4,
                "emb[{i},{j}]: {} vs {want}",
                got.as_f32().unwrap()[i * h + j]
            );
        }
    }

    // ---- full encoder lowers GPU == CPU ----
    let cfg = BertConfig {
        vocab,
        n_heads: 2,
        head_dim: 2,
        intermediate: 6,
        layers: 2,
        max_pos: 8,
        eps,
        gelu_tanh: false, // exact erf GELU (BERT default); also exercises the GeluErf lowering GPU==CPU
    };
    let b2 = Builder::new();
    let tok2 = b2.constant("tok", TensorType::f32(vec![seq]));
    let mk_sq = |bb: &Builder, n: &str| bb.constant(n, TensorType::f32(vec![h, h]));
    let mk_v = |bb: &Builder, n: &str, k: usize| bb.constant(n, TensorType::f32(vec![k]));
    let mut binds: Vec<(poot_graph_ir::ValueId, Vec<usize>)> = Vec::new();
    let reg = |t: poot_graph_ir::builder::Traced, s: Vec<usize>, bd: &mut Vec<_>| {
        bd.push((t.id, s));
        t
    };
    let we2 = reg(
        b2.constant("we", TensorType::f32(vec![vocab, h])),
        vec![vocab, h],
        &mut binds,
    );
    let pe2 = reg(
        b2.constant("pe", TensorType::f32(vec![cfg.max_pos, h])),
        vec![cfg.max_pos, h],
        &mut binds,
    );
    let te2 = reg(
        b2.constant("te", TensorType::f32(vec![2, h])),
        vec![2, h],
        &mut binds,
    );
    let elw = reg(
        b2.constant("elw", TensorType::f32(vec![h])),
        vec![h],
        &mut binds,
    );
    let elb = reg(
        b2.constant("elb", TensorType::f32(vec![h])),
        vec![h],
        &mut binds,
    );
    let mut layers = Vec::new();
    for l in 0..cfg.layers {
        let nm = |s: &str| format!("{s}{l}");
        layers.push(VisionBlockWeights {
            ln1_w: reg(mk_v(&b2, &nm("l1w"), h), vec![h], &mut binds),
            ln1_b: reg(mk_v(&b2, &nm("l1b"), h), vec![h], &mut binds),
            mha: MhaWeights {
                wq: reg(mk_sq(&b2, &nm("wq")), vec![h, h], &mut binds),
                bq: reg(mk_v(&b2, &nm("bq"), h), vec![h], &mut binds),
                wk: reg(mk_sq(&b2, &nm("wk")), vec![h, h], &mut binds),
                bk: reg(mk_v(&b2, &nm("bk"), h), vec![h], &mut binds),
                wv: reg(mk_sq(&b2, &nm("wv")), vec![h, h], &mut binds),
                bv: reg(mk_v(&b2, &nm("bv"), h), vec![h], &mut binds),
                wo: reg(mk_sq(&b2, &nm("wo")), vec![h, h], &mut binds),
                bo: reg(mk_v(&b2, &nm("bo"), h), vec![h], &mut binds),
            },
            ln2_w: reg(mk_v(&b2, &nm("l2w"), h), vec![h], &mut binds),
            ln2_b: reg(mk_v(&b2, &nm("l2b"), h), vec![h], &mut binds),
            fc1_w: reg(
                b2.constant(&nm("f1w"), TensorType::f32(vec![h, cfg.intermediate])),
                vec![h, cfg.intermediate],
                &mut binds,
            ),
            fc1_b: reg(
                mk_v(&b2, &nm("f1b"), cfg.intermediate),
                vec![cfg.intermediate],
                &mut binds,
            ),
            fc2_w: reg(
                b2.constant(&nm("f2w"), TensorType::f32(vec![cfg.intermediate, h])),
                vec![cfg.intermediate, h],
                &mut binds,
            ),
            fc2_b: reg(mk_v(&b2, &nm("f2b"), h), vec![h], &mut binds),
        });
    }
    let ew = BertEncoderWeights {
        word_emb: we2,
        pos_emb: pe2,
        type_emb: te2,
        emb_ln_w: elw,
        emb_ln_b: elb,
        layers,
    };
    let enc = trace_bert_encoder(&b2, tok2, &ew, &cfg, seq);
    let g2 = b2.finish(enc);
    let mut in2 = HashMap::new();
    in2.insert(
        tok2.id,
        HostTensor::f32(vec![seq], toks.iter().map(|&t| t as f32).collect()),
    );
    for (k, (id, s)) in binds.iter().enumerate() {
        let n: usize = s.iter().product();
        in2.insert(*id, HostTensor::f32(s.clone(), r(k * 3 + 1, n)));
    }
    let eval_in2: HashMap<_, Value> = in2
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g2, &eval_in2, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(cpu.shape(), vec![1, seq, h]);
    assert!(cpu.as_f32().unwrap().iter().all(|v| v.is_finite()));
    let _gpu_guard = gpu_lock();
    if let Some((mut gpu, target)) = open_engine_or_skip() {
        let g = run_via_contract(&mut gpu, target, &g2, &in2);
        for (i, (a, c)) in g
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap())
            .enumerate()
        {
            assert!((a - c).abs() <= 1e-4, "encoder gpu elem {i}: {a} vs {c}");
        }
    }
}

#[test]
fn batched_cross_encoder_matches_per_pair() {
    // `trace_cross_encoder_batched` scores N padded (query, doc) pairs in one pass with a key-padding mask and must equal
    // `trace_cross_encoder` per pair (only the [CLS] logit feeds the classifier and [CLS] never attends a padded key). CPU eval
    // only, with two pairs of different lengths so padding is real.
    use poot_models::encoder::{
        BertConfig, declare_cross_encoder_constants, trace_cross_encoder,
        trace_cross_encoder_batched,
    };
    let cfg = BertConfig {
        vocab: 12,
        n_heads: 2,
        head_dim: 2,
        intermediate: 6,
        layers: 2,
        max_pos: 16,
        eps: 1e-12,
        gelu_tanh: false,
    };
    // deterministic, name-keyed weights so the per-pair and batched graphs (separate Builders) share values.
    let mut wmap: HashMap<String, Vec<f32>> = HashMap::new();
    let wdata = |wmap: &mut HashMap<String, Vec<f32>>, name: &str, n: usize| -> Vec<f32> {
        wmap.entry(name.to_string())
            .or_insert_with(|| {
                let seed = name
                    .bytes()
                    .fold(7u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
                (0..n)
                    .map(|i| (((i as u32).wrapping_add(seed)) as f32 * 0.12345).sin() * 0.2)
                    .collect()
            })
            .clone()
    };
    // two pairs: lengths 3 and 5, joint token + segment ids (CLS=pos 0).
    let pairs: [(Vec<u32>, Vec<u32>); 2] = [
        (vec![1, 5, 2], vec![0, 0, 1]),
        (vec![3, 7, 2, 9, 4], vec![0, 1, 1, 1, 1]),
    ];
    let seq = pairs.iter().map(|(t, _)| t.len()).max().unwrap();

    // per-pair reference logits.
    let mut per_pair = Vec::new();
    for (toks, types) in &pairs {
        let s = toks.len();
        let b = Builder::new();
        let tt = TensorType::f32(vec![s]);
        let tok = b.constant("tokens", tt.clone());
        let typ = b.constant("token_types", tt);
        let (_w, binds) = declare_cross_encoder_constants(&b, &cfg);
        let out = trace_cross_encoder(&b, tok, typ, &_w, &cfg, s);
        let g = b.finish(out);
        let mut inp = HashMap::new();
        inp.insert(
            tok.id,
            Value::from(HostTensor::f32(
                vec![s],
                toks.iter().map(|&t| t as f32).collect(),
            )),
        );
        inp.insert(
            typ.id,
            Value::from(HostTensor::f32(
                vec![s],
                types.iter().map(|&t| t as f32).collect(),
            )),
        );
        for (id, name) in &binds {
            let n = g.aval(*id).numel();
            inp.insert(
                *id,
                Value::from(HostTensor::f32(
                    g.aval(*id).shape.clone(),
                    wdata(&mut wmap, name, n),
                )),
            );
        }
        per_pair.push(
            eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .unwrap()
                .as_f32()
                .unwrap()[0],
        );
    }

    // batched forward: pad to `seq`, build the additive key mask.
    let batch = pairs.len();
    let mut tok_flat = vec![0.0f32; batch * seq];
    let mut typ_flat = vec![0.0f32; batch * seq];
    let mut mask = vec![0.0f32; batch * seq];
    for (i, (toks, types)) in pairs.iter().enumerate() {
        for j in 0..seq {
            let off = i * seq + j;
            if j < toks.len() {
                tok_flat[off] = toks[j] as f32;
                typ_flat[off] = types[j] as f32;
            } else {
                mask[off] = -1.0e9;
            }
        }
    }
    let b = Builder::new();
    let tt = TensorType::f32(vec![batch * seq]);
    let tok = b.constant("tokens", tt.clone());
    let typ = b.constant("token_types", tt);
    let mc = b.constant("kmask", TensorType::f32(vec![batch, 1, 1, seq]));
    let (_w, binds) = declare_cross_encoder_constants(&b, &cfg);
    let out = trace_cross_encoder_batched(&b, tok, typ, mc, &_w, &cfg, batch, seq);
    let g = b.finish(out);
    let mut inp = HashMap::new();
    inp.insert(
        tok.id,
        Value::from(HostTensor::f32(vec![batch * seq], tok_flat)),
    );
    inp.insert(
        typ.id,
        Value::from(HostTensor::f32(vec![batch * seq], typ_flat)),
    );
    inp.insert(
        mc.id,
        Value::from(HostTensor::f32(vec![batch, 1, 1, seq], mask)),
    );
    for (id, name) in &binds {
        let n = g.aval(*id).numel();
        inp.insert(
            *id,
            Value::from(HostTensor::f32(
                g.aval(*id).shape.clone(),
                wdata(&mut wmap, name, n),
            )),
        );
    }
    let batched = eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(batched.shape(), vec![batch]);
    for (i, want) in per_pair.iter().enumerate() {
        assert!(
            (batched.as_f32().unwrap()[i] - want).abs() < 1e-3,
            "batched logit {i} = {} vs per-pair {want}",
            batched.as_f32().unwrap()[i]
        );
    }
}

#[test]
fn bert_block_post_norm_wiring_and_executor_equiv() {
    // Post-norm BERT block: LayerNorm(x + attn), LayerNorm(y + mlp). Checks the wiring: (1) zeroing attn out-proj (wo,bo) and
    // fc2 gives attn=mlp=0, so block(x) == LN(LN(x, ln1), ln2) (unlike pre-norm, not x); (2) with real weights, GPU == CPU,
    // finite, and transforms x.
    use poot_graph_ir::ops::layernorm;
    use poot_models::encoder::trace_bert_block;
    use poot_models::vision::{MhaWeights, VisionBlockWeights};
    let (seq, nh, hd, inter, eps) = (2usize, 2usize, 2usize, 6usize, 1e-12f32);
    let h = nh * hd; // 4
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, seq, h]));
    let sq = |n: &str| b.constant(n, TensorType::f32(vec![h, h]));
    let vh = |n: &str, k: usize| b.constant(n, TensorType::f32(vec![k]));
    let w = VisionBlockWeights {
        ln1_w: vh("l1w", h),
        ln1_b: vh("l1b", h),
        mha: MhaWeights {
            wq: sq("wq"),
            bq: vh("bq", h),
            wk: sq("wk"),
            bk: vh("bk", h),
            wv: sq("wv"),
            bv: vh("bv", h),
            wo: sq("wo"),
            bo: vh("bo", h),
        },
        ln2_w: vh("l2w", h),
        ln2_b: vh("l2b", h),
        fc1_w: b.constant("f1w", TensorType::f32(vec![h, inter])),
        fc1_b: vh("f1b", inter),
        fc2_w: b.constant("f2w", TensorType::f32(vec![inter, h])),
        fc2_b: vh("f2b", h),
    };
    let out = trace_bert_block(&b, x, &w, nh, hd, eps, false);
    let g = b.finish(out);

    let r = |seed: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| ((i + seed) as f32 * 0.41).sin() * 0.5)
            .collect()
    };
    let xd = r(1, seq * h);
    let l1w = vec![1.0f32; h];
    let l1b = r(2, h);
    let l2w = vec![1.0f32; h];
    let l2b = r(3, h);
    let build = |zero: bool| {
        let mut m = HashMap::new();
        let put = |m: &mut HashMap<_, _>,
                   t: poot_graph_ir::builder::Traced,
                   s: Vec<usize>,
                   d: Vec<f32>| {
            m.insert(t.id, HostTensor::f32(s, d));
        };
        put(&mut m, x, vec![1, seq, h], xd.clone());
        put(&mut m, w.ln1_w, vec![h], l1w.clone());
        put(&mut m, w.ln1_b, vec![h], l1b.clone());
        put(&mut m, w.mha.wq, vec![h, h], r(4, h * h));
        put(&mut m, w.mha.bq, vec![h], r(5, h));
        put(&mut m, w.mha.wk, vec![h, h], r(6, h * h));
        put(&mut m, w.mha.bk, vec![h], r(7, h));
        put(&mut m, w.mha.wv, vec![h, h], r(8, h * h));
        put(&mut m, w.mha.bv, vec![h], r(9, h));
        let (wo, bo) = if zero {
            (vec![0.0; h * h], vec![0.0; h])
        } else {
            (r(10, h * h), r(11, h))
        };
        put(&mut m, w.mha.wo, vec![h, h], wo);
        put(&mut m, w.mha.bo, vec![h], bo);
        put(&mut m, w.ln2_w, vec![h], l2w.clone());
        put(&mut m, w.ln2_b, vec![h], l2b.clone());
        put(&mut m, w.fc1_w, vec![h, inter], r(12, h * inter));
        put(&mut m, w.fc1_b, vec![inter], r(13, inter));
        let (f2w, f2b) = if zero {
            (vec![0.0; inter * h], vec![0.0; h])
        } else {
            (r(14, inter * h), r(15, h))
        };
        put(&mut m, w.fc2_w, vec![inter, h], f2w);
        put(&mut m, w.fc2_b, vec![h], f2b);
        m
    };

    // (1) post-norm wiring: zeroed sublayers -> block(x) == LN(LN(x, ln1), ln2).
    let zeroed_inputs = build(true);
    let zeroed_eval_inputs: HashMap<_, Value> = zeroed_inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let zeroed = eval(
        &g,
        &zeroed_eval_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    // reference: a separate LN(LN(x)) graph.
    let rb = Builder::new();
    let rx = rb.constant("x", TensorType::f32(vec![1, seq, h]));
    let r1w = rb.constant("l1w", TensorType::f32(vec![h]));
    let r1b = rb.constant("l1b", TensorType::f32(vec![h]));
    let r2w = rb.constant("l2w", TensorType::f32(vec![h]));
    let r2b = rb.constant("l2b", TensorType::f32(vec![h]));
    let ln1 = layernorm(&rb, rx, r1w, r1b, eps);
    let ln2 = layernorm(&rb, ln1, r2w, r2b, eps);
    let refg = rb.finish(ln2);
    let mut rin = HashMap::new();
    rin.insert(
        rx.id,
        Value::from(HostTensor::f32(vec![1, seq, h], xd.clone())),
    );
    rin.insert(r1w.id, Value::from(HostTensor::f32(vec![h], l1w.clone())));
    rin.insert(r1b.id, Value::from(HostTensor::f32(vec![h], l1b.clone())));
    rin.insert(r2w.id, Value::from(HostTensor::f32(vec![h], l2w.clone())));
    rin.insert(r2b.id, Value::from(HostTensor::f32(vec![h], l2b.clone())));
    let want = eval(&refg, &rin, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    for (i, (a, c)) in zeroed
        .as_f32()
        .unwrap()
        .iter()
        .zip(want.as_f32().unwrap())
        .enumerate()
    {
        assert!(
            (a - c).abs() < 1e-5,
            "post-norm wiring elem {i}: {a} vs LN(LN(x)) {c}"
        );
    }

    // (2) real weights: GPU == CPU, finite, transforms x.
    let inputs = build(false);
    let eval_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert!(cpu.as_f32().unwrap().iter().all(|v| v.is_finite()));
    assert!(
        cpu.as_f32()
            .unwrap()
            .iter()
            .zip(&xd)
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "block must transform x"
    );
    let _gpu_guard = gpu_lock();
    if let Some((mut gpu, target)) = open_engine_or_skip() {
        let got = run_via_contract(&mut gpu, target, &g, &inputs);
        for (i, (a, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap())
            .enumerate()
        {
            assert!((a - c).abs() <= 1e-4, "bert block gpu elem {i}: {a} vs {c}");
        }
    }
}

#[test]
fn vision_block_residual_wiring_and_executor_equiv() {
    // SigLIP pre-norm block with two residuals. Checks the wiring: (1) zeroing attention out-proj (wo,bo) and fc2 makes the block
    // an exact residual passthrough, block(x) == x; (2) with real weights it lowers GPU == CPU, is finite, and changes x.
    use poot_models::vision::{MhaWeights, VisionBlockWeights, trace_vision_block};
    let (seq, nh, hd, inter) = (2usize, 2usize, 2usize, 6usize);
    let h = nh * hd; // 4
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, seq, h]));
    let sq = |n: &str| b.constant(n, TensorType::f32(vec![h, h]));
    let vh = |n: &str, k: usize| b.constant(n, TensorType::f32(vec![k]));
    let w = VisionBlockWeights {
        ln1_w: vh("l1w", h),
        ln1_b: vh("l1b", h),
        mha: MhaWeights {
            wq: sq("wq"),
            bq: vh("bq", h),
            wk: sq("wk"),
            bk: vh("bk", h),
            wv: sq("wv"),
            bv: vh("bv", h),
            wo: sq("wo"),
            bo: vh("bo", h),
        },
        ln2_w: vh("l2w", h),
        ln2_b: vh("l2b", h),
        fc1_w: b.constant("f1w", TensorType::f32(vec![h, inter])),
        fc1_b: vh("f1b", inter),
        fc2_w: b.constant("f2w", TensorType::f32(vec![inter, h])),
        fc2_b: vh("f2b", h),
    };
    let out = trace_vision_block(&b, x, &w, nh, hd, 1e-5);
    let g = b.finish(out);

    let r = |seed: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| ((i + seed) as f32 * 0.41).sin() * 0.5)
            .collect()
    };
    let xd = r(1, seq * h);
    // helper to assemble the input map; `zero_attn_mlp` zeroes wo/bo/fc2 for the passthrough check.
    let build_inputs = |zero_attn_mlp: bool| {
        let mut m = HashMap::new();
        let put = |m: &mut HashMap<_, _>,
                   t: poot_graph_ir::builder::Traced,
                   shape: Vec<usize>,
                   d: Vec<f32>| {
            m.insert(t.id, HostTensor::f32(shape, d));
        };
        put(&mut m, x, vec![1, seq, h], xd.clone());
        put(&mut m, w.ln1_w, vec![h], vec![1.0; h]);
        put(&mut m, w.ln1_b, vec![h], r(2, h));
        put(&mut m, w.mha.wq, vec![h, h], r(3, h * h));
        put(&mut m, w.mha.bq, vec![h], r(4, h));
        put(&mut m, w.mha.wk, vec![h, h], r(5, h * h));
        put(&mut m, w.mha.bk, vec![h], r(6, h));
        put(&mut m, w.mha.wv, vec![h, h], r(7, h * h));
        put(&mut m, w.mha.bv, vec![h], r(8, h));
        let (wo, bo) = if zero_attn_mlp {
            (vec![0.0; h * h], vec![0.0; h])
        } else {
            (r(9, h * h), r(10, h))
        };
        put(&mut m, w.mha.wo, vec![h, h], wo);
        put(&mut m, w.mha.bo, vec![h], bo);
        put(&mut m, w.ln2_w, vec![h], vec![1.0; h]);
        put(&mut m, w.ln2_b, vec![h], r(11, h));
        put(&mut m, w.fc1_w, vec![h, inter], r(12, h * inter));
        put(&mut m, w.fc1_b, vec![inter], r(13, inter));
        let (f2w, f2b) = if zero_attn_mlp {
            (vec![0.0; inter * h], vec![0.0; h])
        } else {
            (r(14, inter * h), r(15, h))
        };
        put(&mut m, w.fc2_w, vec![inter, h], f2w);
        put(&mut m, w.fc2_b, vec![h], f2b);
        m
    };

    // (1) residual passthrough.
    let passthrough_inputs = build_inputs(true);
    let passthrough_eval_inputs: HashMap<_, Value> = passthrough_inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let passthrough = eval(
        &g,
        &passthrough_eval_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    for (i, (got, want)) in passthrough.as_f32().unwrap().iter().zip(&xd).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "residual passthrough elem {i}: {got} vs {want}"
        );
    }

    // (2) real weights: finite, changes x, and GPU == CPU.
    let inputs = build_inputs(false);
    let eval_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(cpu.shape(), vec![1, seq, h]);
    assert!(
        cpu.as_f32().unwrap().iter().all(|v| v.is_finite()),
        "block output not finite"
    );
    let changed = cpu
        .as_f32()
        .unwrap()
        .iter()
        .zip(&xd)
        .any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(changed, "block must transform x (sublayers contribute)");
    let _gpu_guard = gpu_lock();
    if let Some((mut gpu, target)) = open_engine_or_skip() {
        let got = run_via_contract(&mut gpu, target, &g, &inputs);
        for (i, (a, c)) in got
            .as_f32()
            .unwrap()
            .iter()
            .zip(cpu.as_f32().unwrap())
            .enumerate()
        {
            // a deep composition near zero: absolute FP-accumulation tolerance (GPU vs CPU op order).
            assert!((a - c).abs() <= 1e-4, "block gpu elem {i}: {a} vs {c}");
        }
    }
}

#[test]
fn image_splice_replaces_placeholder_embeddings() {
    // `trace_image_splice` replaces the <image> positions' text embeddings with the visual tokens (in order) and keeps the rest, on CPU and GPU.
    use poot_models::vision::{image_splice_inverse_map, trace_image_splice};
    let (seq, hidden, n_vis) = (5usize, 3usize, 2usize);
    let b = Builder::new();
    let text = b.constant("t", TensorType::f32(vec![seq, hidden]));
    let vis = b.constant("v", TensorType::f32(vec![n_vis, hidden]));
    let inv = b.constant("inv", TensorType::f32(vec![seq]));
    let out = trace_image_splice(&b, text, vis, inv);
    let (ti, vi, ii) = (text.id, vis.id, inv.id);
    let g = b.finish(out);

    // image tokens (id 7) at positions 1 and 3.
    let tokens = [4u32, 7, 9, 7, 2];
    let inv_map = image_splice_inverse_map(&tokens, 7); // [-1, 0, -1, 1, -1]
    let textd: Vec<f32> = (0..seq * hidden).map(|i| i as f32).collect();
    let visd: Vec<f32> = (0..n_vis * hidden).map(|i| 100.0 + i as f32).collect();
    let mut inputs = HashMap::new();
    inputs.insert(ti, HostTensor::f32(vec![seq, hidden], textd.clone()));
    inputs.insert(vi, HostTensor::f32(vec![n_vis, hidden], visd.clone()));
    inputs.insert(ii, HostTensor::f32(vec![seq], inv_map.clone()));

    // reference: position p gets visual[inv[p]] if inv[p]>=0 else text[p].
    let mut want = textd.clone();
    for p in 0..seq {
        let k = inv_map[p];
        if k >= 0.0 {
            let k = k as usize;
            want[p * hidden..(p + 1) * hidden].copy_from_slice(&visd[k * hidden..(k + 1) * hidden]);
        }
    }

    let eval_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(cpu.shape(), vec![seq, hidden]);
    assert_eq!(cpu.as_f32().unwrap(), &want[..], "cpu splice");
    let _gpu_guard = gpu_lock();
    if let Some((mut gpu, target)) = open_engine_or_skip() {
        let got = run_via_contract(&mut gpu, target, &g, &inputs);
        for (i, (a, w)) in got.as_f32().unwrap().iter().zip(&want).enumerate() {
            assert!((a - w).abs() < 1e-5, "gpu splice elem {i}: {a} vs {w}");
        }
    }
}

#[test]
fn connector_pixel_shuffle_matches_reference() {
    // The idefics3 pixel shuffle regroups [1, seq, embed] -> [1, seq/s^2, embed*s^2] by a view/permute sequence;
    // `trace_connector` adds a linear projector. With an identity projector the output is the shuffle, checked against a hand
    // reference (same permutes via 3D transposes) on CPU and GPU.
    use poot_models::vision::trace_connector;
    let (embed, h, s) = (2usize, 4usize, 2usize); // seq 16, out_seq 4, out_embed embed*s^2 = 8.
    let seq = h * h;
    let oe = embed * s * s;
    let out_seq = seq / (s * s);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, seq, embed]));
    let proj = b.constant("p", TensorType::f32(vec![oe, oe])); // identity -> passthrough.
    let out = trace_connector(&b, x, proj, embed, s);
    let (xi, pi) = (x.id, proj.id);
    let g = b.finish(out);

    let xd: Vec<f32> = (0..seq * embed).map(|i| i as f32).collect();
    let identity: Vec<f32> = (0..oe * oe)
        .map(|i| if i / oe == i % oe { 1.0 } else { 0.0 })
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![1, seq, embed], xd.clone()));
    inputs.insert(pi, HostTensor::f32(vec![oe, oe], identity));

    // hand reference: [h, w/s, embed*s] -(perm 1,0,2)-> [w/s, h, embed*s] -(reshape)-(perm 1,0,2)->
    // [h/s, w/s, embed*s*s] == [out_seq, oe].
    fn t3(inp: &[f32], d0: usize, d1: usize, d2: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; d0 * d1 * d2];
        for i0 in 0..d0 {
            for i1 in 0..d1 {
                for i2 in 0..d2 {
                    out[i1 * d0 * d2 + i0 * d2 + i2] = inp[i0 * d1 * d2 + i1 * d2 + i2];
                }
            }
        }
        out
    }
    let (w, es) = (h, embed * s);
    let a = t3(&xd, h, w / s, es); // [w/s, h, embed*s]
    let want = t3(&a, w / s, h / s, oe); // [h/s, w/s, oe] flat == [out_seq, oe]

    let eval_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(cpu.shape(), vec![1, out_seq, oe]);
    for (i, (got, w)) in cpu.as_f32().unwrap().iter().zip(&want).enumerate() {
        assert!((got - w).abs() < 1e-5, "cpu shuffle elem {i}: {got} vs {w}");
    }
    let _gpu_guard = gpu_lock();
    if let Some((mut gpu, target)) = open_engine_or_skip() {
        let got = run_via_contract(&mut gpu, target, &g, &inputs);
        for (i, (a, w)) in got.as_f32().unwrap().iter().zip(&want).enumerate() {
            assert!((a - w).abs() < 1e-4, "gpu shuffle elem {i}: {a} vs {w}");
        }
    }
}

#[test]
fn vision_mha_matches_reference() {
    // Multi-head self-attention assembly (qkv proj, head split, full non-causal attention, head merge, out proj) against a hand
    // reference on a tiny case (seq 3, 2 heads, head_dim 2), on CPU and GPU.
    use poot_models::vision::{MhaWeights, trace_vision_mha};
    let (seq, nh, hd) = (3usize, 2usize, 2usize);
    let h = nh * hd; // 4
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, seq, h]));
    let mk = |n: &str, r: usize, c: usize| b.constant(n, TensorType::f32(vec![r, c]));
    let w = MhaWeights {
        wq: mk("wq", h, h),
        bq: b.constant("bq", TensorType::f32(vec![h])),
        wk: mk("wk", h, h),
        bk: b.constant("bk", TensorType::f32(vec![h])),
        wv: mk("wv", h, h),
        bv: b.constant("bv", TensorType::f32(vec![h])),
        wo: mk("wo", h, h),
        bo: b.constant("bo", TensorType::f32(vec![h])),
    };
    let out = trace_vision_mha(&b, x, &w, nh, hd);
    let ids: Vec<_> = [
        x.id, w.wq.id, w.bq.id, w.wk.id, w.bk.id, w.wv.id, w.bv.id, w.wo.id, w.bo.id,
    ]
    .to_vec();
    let g = b.finish(out);

    // deterministic data.
    let r = |seed: usize, n: usize| -> Vec<f32> {
        (0..n).map(|i| ((i + seed) as f32 * 0.37).sin()).collect()
    };
    let xd = r(1, seq * h);
    let (wqd, wkd, wvd, wod) = (r(2, h * h), r(3, h * h), r(4, h * h), r(5, h * h));
    let (bqd, bkd, bvd, bod) = (r(6, h), r(7, h), r(8, h), r(9, h));
    let mut inputs = HashMap::new();
    for (id, d, shape) in [
        (ids[0], &xd, vec![1, seq, h]),
        (ids[1], &wqd, vec![h, h]),
        (ids[2], &bqd, vec![h]),
        (ids[3], &wkd, vec![h, h]),
        (ids[4], &bkd, vec![h]),
        (ids[5], &wvd, vec![h, h]),
        (ids[6], &bvd, vec![h]),
        (ids[7], &wod, vec![h, h]),
        (ids[8], &bod, vec![h]),
    ] {
        inputs.insert(id, HostTensor::f32(shape, d.clone()));
    }

    // ---- hand reference ----
    let matmul = |a: &[f32], ar: usize, ac: usize, bb: &[f32], bc: usize| -> Vec<f32> {
        let mut o = vec![0.0f32; ar * bc];
        for i in 0..ar {
            for j in 0..bc {
                let mut s = 0.0;
                for k in 0..ac {
                    s += a[i * ac + k] * bb[k * bc + j];
                }
                o[i * bc + j] = s;
            }
        }
        o
    };
    let add_bias = |m: &mut [f32], bias: &[f32], rows: usize, cols: usize| {
        for i in 0..rows {
            for j in 0..cols {
                m[i * cols + j] += bias[j];
            }
        }
    };
    let mut q = matmul(&xd, seq, h, &wqd, h);
    add_bias(&mut q, &bqd, seq, h);
    let mut k = matmul(&xd, seq, h, &wkd, h);
    add_bias(&mut k, &bkd, seq, h);
    let mut v = matmul(&xd, seq, h, &wvd, h);
    add_bias(&mut v, &bvd, seq, h);
    let scale = 1.0 / (hd as f32).sqrt();
    // per-head attention -> attn[seq, h] (head ho occupies cols [ho*hd, ho*hd+hd)).
    let mut attn = vec![0.0f32; seq * h];
    for ho in 0..nh {
        let col = ho * hd;
        for i in 0..seq {
            // scores over j, softmax.
            let mut sc = vec![0.0f32; seq];
            for j in 0..seq {
                let mut d = 0.0;
                for t in 0..hd {
                    d += q[i * h + col + t] * k[j * h + col + t];
                }
                sc[j] = d * scale;
            }
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0.0;
            for s in sc.iter_mut() {
                *s = (*s - mx).exp();
                den += *s;
            }
            for t in 0..hd {
                let mut acc = 0.0;
                for j in 0..seq {
                    acc += (sc[j] / den) * v[j * h + col + t];
                }
                attn[i * h + col + t] = acc;
            }
        }
    }
    let mut want = matmul(&attn, seq, h, &wod, h);
    add_bias(&mut want, &bod, seq, h);

    let eval_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(cpu.shape(), vec![1, seq, h]);
    for (i, (got, w)) in cpu.as_f32().unwrap().iter().zip(&want).enumerate() {
        assert!((got - w).abs() < 1e-5, "cpu mha elem {i}: {got} vs {w}");
    }
    let _gpu_guard = gpu_lock();
    if let Some((mut gpu, target)) = open_engine_or_skip() {
        let got = run_via_contract(&mut gpu, target, &g, &inputs);
        for (i, (a, w)) in got.as_f32().unwrap().iter().zip(&want).enumerate() {
            assert!((a - w).abs() < 1e-4, "gpu mha elem {i}: {a} vs {w}");
        }
    }
}

#[test]
fn patch_embed_im2col_matches_reference() {
    // `trace_patch_embed` splits a [C,S,S] image into patches in [c,i,j] order (the conv2d weight layout) and projects them.
    // With identity weight and zero bias/pos the output is the im2col patch matrix, verifying the reshape/transpose
    // extraction. CPU and (if the 5D transpose lowers) GPU.
    use poot_models::vision::trace_patch_embed;
    let (c, s, p) = (2usize, 4usize, 2usize); // grid 2, num_patches 4, patch_dim 8.
    let grid = s / p;
    let np = grid * grid;
    let pd = c * p * p;
    let b = Builder::new();
    let pixels = b.constant("pix", TensorType::f32(vec![c, s, s]));
    let weight = b.constant("w", TensorType::f32(vec![pd, pd])); // identity [8,8].
    let bias = b.constant("b", TensorType::f32(vec![pd]));
    let pos = b.constant("pos", TensorType::f32(vec![np, pd]));
    let out = trace_patch_embed(&b, pixels, weight, bias, pos, c, s, p);
    let (pixi, wi, bi, posi) = (pixels.id, weight.id, bias.id, pos.id);
    let g = b.finish(out);

    // pixel value encodes its (c, y, x) so the reference patch order is checkable.
    let mut pix = vec![0.0f32; c * s * s];
    for ch in 0..c {
        for y in 0..s {
            for x in 0..s {
                pix[ch * s * s + y * s + x] = (ch * 100 + y * 10 + x) as f32;
            }
        }
    }
    let identity: Vec<f32> = (0..pd * pd)
        .map(|i| if i / pd == i % pd { 1.0 } else { 0.0 })
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(pixi, HostTensor::f32(vec![c, s, s], pix.clone()));
    inputs.insert(wi, HostTensor::f32(vec![pd, pd], identity));
    inputs.insert(bi, HostTensor::f32(vec![pd], vec![0.0; pd]));
    inputs.insert(posi, HostTensor::f32(vec![np, pd], vec![0.0; np * pd]));

    // reference im2col: patch (py,px) row-major; within a patch, [c, i, j] order.
    let mut want = vec![0.0f32; np * pd];
    for py in 0..grid {
        for px in 0..grid {
            let patch = py * grid + px;
            let mut k = 0;
            for ch in 0..c {
                for i in 0..p {
                    for j in 0..p {
                        want[patch * pd + k] = pix[ch * s * s + (py * p + i) * s + (px * p + j)];
                        k += 1;
                    }
                }
            }
        }
    }

    let eval_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(cpu.shape(), vec![np, pd]);
    for (i, (got, w)) in cpu.as_f32().unwrap().iter().zip(&want).enumerate() {
        assert!((got - w).abs() < 1e-5, "cpu patch elem {i}: {got} vs {w}");
    }

    let _gpu_guard = gpu_lock();
    if let Some((mut gpu, target)) = open_engine_or_skip() {
        let got = run_via_contract(&mut gpu, target, &g, &inputs);
        assert_eq!(got.shape(), vec![np, pd]);
        for (i, (a, w)) in got.as_f32().unwrap().iter().zip(&want).enumerate() {
            assert!((a - w).abs() < 1e-4, "gpu patch elem {i}: {a} vs {w}");
        }
    } else {
        eprintln!("no GPU; CPU-only patch-embed check");
    }
}
