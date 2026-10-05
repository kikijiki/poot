//! Local diagnostic: dump a GGUF's architecture metadata and every tensor name matching a filter,
//! via `GgufIndex::open` (header only, so it works on a partial download such as the first
//! 256MB fetched with an HTTP Range request). Written for the Mixtral-8x7B coherence work
//! (docs/updates/, search "mixtral-8x7b"): some pre-2024 conversions use per-expert
//! `ffn_gate.{e}.weight` tensors that poot's loader does not recognize, while current llama.cpp
//! conversions use merged 3D `ffn_gate_exps.weight`.
//! Usage:
//!   cargo run -p poot-load --example gguf_dump_tensors -- <path> [name-substring-filter]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: gguf_dump_tensors <path> [filter]");
    let filter = args.get(2).cloned();
    let g = poot_load::gguf::GgufIndex::open(path).expect("open gguf header");
    println!("architecture = {:?}", g.architecture());
    for key in [
        "llama.expert_count",
        "llama.expert_used_count",
        "llama.feed_forward_length",
        "llama.block_count",
    ] {
        if let Some(v) = g.get(key) {
            println!("{key} = {v:?}");
        }
    }
    let mut names: Vec<&String> = g.tensors.keys().collect();
    names.sort();
    println!("total tensors in header window: {}", names.len());
    for n in &names {
        if let Some(f) = &filter
            && !n.contains(f.as_str())
        {
            continue;
        }
        let info = &g.tensors[*n];
        println!("{n}  type={} dims={:?}", info.ggml_type, info.dims);
    }
}
