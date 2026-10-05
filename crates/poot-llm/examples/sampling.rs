//! Next-token sampling: greedy (argmax) vs temperature + top-k + top-p (nucleus), with a deterministic
//! per-seed RNG (no `rand` dep). `poot-serve` exposes these as the request's {temperature, top_k, top_p, seed}.
//!
//! Run: `cargo run -p poot-llm --example sampling` (CPU only).

use poot_llm::Sampler;

fn main() -> Result<(), poot_llm::SamplerFault> {
    // toy logits over a 5-token vocab; token 1 is the argmax.
    let logits = [0.1f32, 3.0, 0.5, 2.9, 1.2];

    let mut greedy = Sampler::greedy();
    println!("greedy (temperature 0) -> token {}", greedy.pick(&logits)?);

    // temperature sampling is deterministic for a given seed; different seeds explore differently.
    for seed in [1u64, 2, 3] {
        let mut s = Sampler::new(0.8, 0, 0.95, seed);
        let picks: Vec<usize> = (0..8).map(|_| s.pick(&logits)).collect::<Result<_, _>>()?;
        println!("temp=0.8 top_p=0.95 seed={seed} -> {picks:?}");
    }

    // top_k=1 keeps only the max logit, so the draw is forced to the argmax regardless of temperature.
    let mut k1 = Sampler::new(1.5, 1, 1.0, 7);
    let forced: Vec<usize> = (0..8).map(|_| k1.pick(&logits)).collect::<Result<_, _>>()?;
    println!("temp=1.5 top_k=1 -> {forced:?} (all argmax)");
    Ok(())
}
