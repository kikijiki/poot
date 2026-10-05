use crate::core::runner::Runner;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    (d / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

// Localize the llama3.2 GGUF divergence: load the GGUF and the safetensors (known-good) llama-3.2-1b and
// compare each weight tensor by cosine. Both loaders key the map by the same HF names, so a low cosine names
// the mis-dequantized or mis-mapped tensor. Run with --release --ignored --nocapture.
#[test]
#[ignore = "differential: loads two 1B llama-3.2 models (gguf + safetensors)"]
fn gguf_vs_safetensors_weights() {
    // To vet another quant (e.g. Q5_K_M), point this at that file under POOT_MODELS_DIR.
    let Some(gguf) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    )) else {
        return;
    };
    let Some(st) = poot_test_util::model_path(poot_test_util::checkpoint!("llama-3.2-1b-instruct"))
    else {
        return;
    };
    let rg = Runner::load_gguf(&gguf).expect("gguf");
    let rs = Runner::load(&st).expect("safetensors");
    let mut keys: Vec<&String> = rg.weights.keys().collect();
    keys.sort();
    for k in keys {
        let Some(b) = rs.weights.get(k) else {
            eprintln!(
                "{k}: MISSING in safetensors (gguf shape {:?})",
                rg.weights[k].as_host().expect("dense weight").shape()
            );
            continue;
        };
        let a = &rg.weights[k];
        let cos = cosine(
            a.as_host().expect("dense weight").as_f32().unwrap(),
            b.as_host().expect("dense weight").as_f32().unwrap(),
        );
        let shape_mismatch = if a.as_host().expect("dense weight").shape()
            != b.as_host().expect("dense weight").shape()
        {
            format!(
                "  SHAPE g={:?} s={:?}",
                a.as_host().expect("dense weight").shape(),
                b.as_host().expect("dense weight").shape()
            )
        } else {
            String::new()
        };
        if cos < 0.95 || !shape_mismatch.is_empty() {
            eprintln!("{k}: cos={cos:.4}{shape_mismatch}");
        }
    }
    // also report any safetensors keys missing from gguf
    for k in rs.weights.keys() {
        if !rg.weights.contains_key(k) {
            eprintln!(
                "{k}: MISSING in gguf (safetensors shape {:?})",
                rs.weights[k].as_host().expect("dense weight").shape()
            );
        }
    }
    eprintln!("(only tensors with cos<0.95 or shape mismatch shown; silence above = all match)");
}
