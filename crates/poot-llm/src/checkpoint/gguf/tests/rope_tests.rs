//! The Runner's RoPE table over HF `rope_scaling` (the flavor helpers and their tests live with the formula
//! in `poot_graph_ir::rope_table`, Card 562a).

use poot_load::RopeScaling;
use poot_models::qwen2::Qwen2Config;

fn llama3() -> RopeScaling {
    RopeScaling {
        rope_type: "llama3".to_string(),
        factor: 32.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_position_embeddings: 8192,
        long_factor: None,
        short_factor: None,
        beta_fast: None,
        beta_slow: None,
        attention_factor: None,
        mscale: None,
        mscale_all_dim: None,
    }
}

// LongRoPE long-context regime

fn longrope(orig: usize, short: Vec<f32>, long: Option<Vec<f32>>) -> RopeScaling {
    RopeScaling {
        rope_type: "longrope".to_string(),
        original_max_position_embeddings: orig,
        short_factor: Some(short),
        long_factor: long,
        ..llama3()
    }
}

#[test]
fn longrope_long_regime_end_to_end_rescales_every_row_including_early_positions() {
    // Once the table's capacity crosses the original context, every row (including early positions) uses
    // long_factor, not just rows past the threshold. This deliberately deviates from a per-row switch (see
    // `poot_graph_ir::rope_table::RopeFlavor::LongRope`) and the test pins it against a per-row refactor.
    let mut cfg = Qwen2Config::qwen2_0_5b();
    cfg.rotary_dim = 8; // half = 4
    cfg.max_pos = 64; // capacity > orig below -> long regime active for the WHOLE table
    let theta = 10_000.0f32;
    let orig = 16usize;
    let short = vec![1.0f32; 4]; // identity: no rescale
    let long = vec![2.0f32; 4]; // halves inv_freq
    let s = longrope(orig, short, Some(long));

    let (cos, sin) = crate::checkpoint::gguf::rope::rope_tables(&cfg, theta, Some(&s), None);
    let d = cfg.rotary_dim;
    let half = d / 2;

    // The constant attention magnitude factor (same formula for either regime, see rope_tables' longrope
    // branch): af = sqrt(1 + ln(p/orig) / ln(orig)) since p/orig = 64/16 = 4.0 > 1.
    let (p64, orig64) = (cfg.max_pos as f64, orig as f64);
    let af64 = (1.0 + (p64 / orig64).ln() / orig64.ln()).sqrt();

    // Independent f64 reference: every position (including pos=1, well below orig=16) uses the
    // long_factor=2.0 divisor, i.e. base^(-2j/d) / 2.0, with the attention factor on top, as for short-regime
    // LongRoPE.
    for &pos in &[0usize, 1, 15, 16, 17, 63] {
        for j in 0..half {
            let base64 = (theta as f64).powf(-((2 * j) as f64) / d as f64);
            let ang64 = pos as f64 * (base64 / 2.0);
            let (c64, s64) = (ang64.cos() * af64, ang64.sin() * af64);
            let c_got = cos.as_f32().unwrap()[pos * d + j] as f64;
            let s_got = sin.as_f32().unwrap()[pos * d + j] as f64;
            assert!(
                (c_got - c64).abs() < 1e-4 && (s_got - s64).abs() < 1e-4,
                "pos={pos} j={j}: got cos={c_got} sin={s_got}, want cos={c64} sin={s64} \
                     (long_factor must apply to every row, including pos < original_max_position)"
            );
        }
    }
}
