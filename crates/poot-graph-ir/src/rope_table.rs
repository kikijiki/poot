//! RoPE cos/sin tables: the one formula behind the Runner's host `rope.cos`/`rope.sin` constants and
//! [`ComputedConst::Rope`](crate::ComputedConst::Rope) (Card 562a, R-562-2).
//!
//! A table is `[positions, dim]` row-major with its two halves duplicated (the half-split rotary
//! decomposition: row `p`, column `i` is `cos(p * inv_freq[i % (dim / 2)])`, times the flavor's
//! attention factor). `dim` is the rotary width, which is the head dim except for partial rotary.
//!
//! The scaled flavors follow HF `transformers`' `modeling_rope_utils`. Two of them depend on a sequence
//! length that HF reads per forward call (dynamic NTK's base, LongRoPE's regime); a static table stands
//! in with its own `positions`, the largest position it can serve.

use crate::graph::ComputedConst;

/// How a table's inverse frequencies are derived from the base. Every flavor is fully determined by
/// a checkpoint config; [`rope_tables`] also takes GGUF's per-frequency factors separately.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeFlavor<'a> {
    /// `inv_freq[j] = theta^(-2j / dim)`.
    Plain,
    /// Position interpolation (HF `linear`): every inverse frequency divided by `factor`
    /// (`pos / factor * f == pos * (f / factor)`). A `factor <= 0` leaves the table plain.
    Linear { factor: f32 },
    /// Dynamic NTK (HF `dynamic`): once `positions` exceeds `original`, the base grows as
    /// `theta * (factor * positions / original - (factor - 1))^(dim / (dim - 2))`.
    DynamicNtk { factor: f32, original: usize },
    /// YaRN (HF `yarn`): a ramp over the correction-dimension range from `beta_fast`/`beta_slow`
    /// blends the unscaled and the `factor`-interpolated frequency, and an attention factor scales the
    /// table (`attention_factor`, or HF's `get_mscale(factor)` when `None`).
    Yarn {
        factor: f32,
        original: usize,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: Option<f32>,
    },
    /// Llama 3 NTK-by-parts (HF `llama3`): short wavelengths kept, long ones divided by `factor`, the
    /// band between blended.
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original: usize,
    },
    /// LongRoPE (HF `longrope`, phi3): each inverse frequency divided by a per-frequency factor, the
    /// `long` regime when `positions` exceeds `original` (and `long` is present), else `short`; the
    /// table is scaled by `sqrt(1 + ln(max_positions / original) / ln(original))` when
    /// `max_positions > original`, which HF computes once from the config.
    LongRope {
        original: usize,
        max_positions: usize,
        short: Option<&'a [f32]>,
        long: Option<&'a [f32]>,
    },
}

/// A `[positions, dim]` cos table and sin table, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeTables {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

/// Build the cos/sin tables. `freq_factors` are GGUF's baked per-frequency divisors
/// (`rope_freqs.weight`: llama.cpp folds llama3's rescale into the checkpoint as
/// `inv_freq[j] / factors[j]`); they apply to every flavor but YaRN and a LongRoPE flavor that has
/// factors of its own.
pub fn rope_tables(
    dim: usize,
    positions: usize,
    theta: f32,
    flavor: &RopeFlavor<'_>,
    freq_factors: Option<&[f32]>,
) -> RopeTables {
    let d = dim;
    let half = d / 2;
    let p = positions;
    let (long_short, longrope_af) = match *flavor {
        RopeFlavor::LongRope {
            original,
            max_positions,
            short,
            long,
        } => {
            let orig = original.max(1) as f32;
            let ratio = max_positions as f32 / orig;
            let af = if ratio > 1.0 {
                (1.0 + ratio.ln() / orig.ln()).sqrt()
            } else {
                1.0
            };
            let factors = if p > original.max(1) && long.is_some() {
                long
            } else {
                short
            };
            (factors, af)
        }
        _ => (None, 1.0),
    };
    let (inv_freq, attn_factor): (Vec<f32>, f32) = match *flavor {
        RopeFlavor::Yarn {
            factor,
            original,
            beta_fast,
            beta_slow,
            attention_factor,
        } => yarn_inv_freq(
            theta,
            d,
            factor,
            original,
            beta_fast,
            beta_slow,
            attention_factor,
        ),
        _ => {
            let base_theta = dynamic_ntk_base(theta, flavor, p, d);
            let inv_freq = (0..half)
                .map(|j| {
                    let base = base_theta.powf(-((2 * j) as f32) / d as f32);
                    if let Some(ls) = long_short {
                        return base / ls.get(j).copied().unwrap_or(1.0);
                    }
                    let f = llama3_rescale(base, flavor);
                    let f = linear_rescale(f, flavor);
                    match freq_factors {
                        Some(ff) if j < ff.len() => f / ff[j],
                        _ => f,
                    }
                })
                .collect();
            (inv_freq, longrope_af)
        }
    };
    let mut cos = vec![0.0f32; p * d];
    let mut sin = vec![0.0f32; p * d];
    for pos in 0..p {
        for j in 0..half {
            let ang = pos as f32 * inv_freq[j];
            let (c, s) = (ang.cos() * attn_factor, ang.sin() * attn_factor);
            cos[pos * d + j] = c;
            cos[pos * d + half + j] = c;
            sin[pos * d + j] = s;
            sin[pos * d + half + j] = s;
        }
    }
    RopeTables { cos, sin }
}

/// Llama 3 rescale of one inverse frequency; any other flavor passes it through.
fn llama3_rescale(inv_freq: f32, flavor: &RopeFlavor<'_>) -> f32 {
    let RopeFlavor::Llama3 {
        factor,
        low_freq_factor,
        high_freq_factor,
        original,
    } = *flavor
    else {
        return inv_freq;
    };
    let old_ctx = original as f32;
    let low_wavelen = old_ctx / low_freq_factor;
    let high_wavelen = old_ctx / high_freq_factor;
    let wavelen = 2.0 * std::f32::consts::PI / inv_freq;
    if wavelen < high_wavelen {
        inv_freq
    } else if wavelen > low_wavelen {
        inv_freq / factor
    } else {
        let smooth = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
        (1.0 - smooth) * inv_freq / factor + smooth * inv_freq
    }
}

/// Linear interpolation of one inverse frequency; any other flavor, or `factor <= 0`, passes it
/// through.
fn linear_rescale(inv_freq: f32, flavor: &RopeFlavor<'_>) -> f32 {
    match *flavor {
        RopeFlavor::Linear { factor } if factor > 0.0 => inv_freq / factor,
        _ => inv_freq,
    }
}

/// Dynamic NTK's base for a table of `seq_len` positions; any other flavor, or `seq_len` within the
/// original context, keeps `theta`.
fn dynamic_ntk_base(theta: f32, flavor: &RopeFlavor<'_>, seq_len: usize, dim: usize) -> f32 {
    let RopeFlavor::DynamicNtk { factor, original } = *flavor else {
        return theta;
    };
    let orig = original.max(1) as f32;
    let seq_len = seq_len as f32;
    if seq_len <= orig {
        return theta;
    }
    let dim = dim as f32;
    theta * ((factor * seq_len / orig) - (factor - 1.0)).powf(dim / (dim - 2.0))
}

/// YaRN's attention (temperature) factor: `attention_factor`, or HF's default `get_mscale(factor)`
/// (`1` at `factor <= 1`, else `0.1 * ln(factor) + 1`).
fn yarn_attention_factor(factor: f32, attention_factor: Option<f32>) -> f32 {
    let factor = factor.max(1e-6);
    attention_factor.unwrap_or(if factor <= 1.0 {
        1.0
    } else {
        0.1 * factor.ln() + 1.0
    })
}

/// YaRN's inverse frequencies and attention factor (HF `_compute_yarn_parameters`).
fn yarn_inv_freq(
    theta: f32,
    d: usize,
    factor: f32,
    original: usize,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: Option<f32>,
) -> (Vec<f32>, f32) {
    let half = d / 2;
    let factor = factor.max(1e-6);
    let orig = original.max(1) as f32;
    // find_correction_dim: the frequency index at which a wavelength of `num_rotations` full turns
    // over the original context falls.
    let correction_dim = |num_rotations: f32| -> f32 {
        (d as f32 * (orig / (num_rotations * 2.0 * std::f32::consts::PI)).ln()) / (2.0 * theta.ln())
    };
    let low = correction_dim(beta_fast).floor().max(0.0);
    let high = correction_dim(beta_slow).ceil().min((d - 1) as f32);
    let (low, high) = if low == high {
        (low, high + 0.001)
    } else {
        (low, high)
    };
    let attention_factor = yarn_attention_factor(factor, attention_factor);
    let inv_freq = (0..half)
        .map(|j| {
            let pos_freq = theta.powf((2 * j) as f32 / d as f32);
            let extrap = 1.0 / pos_freq;
            let interp = 1.0 / (factor * pos_freq);
            let ramp = ((j as f32 - low) / (high - low)).clamp(0.0, 1.0);
            let extrap_factor = 1.0 - ramp;
            interp * (1.0 - extrap_factor) + extrap * extrap_factor
        })
        .collect();
    (inv_freq, attention_factor)
}

/// Which table of a RoPE pair a [`ComputedConst::Rope`] materializes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopeTable {
    Cos,
    Sin,
}

/// A [`RopeFlavor`] whose parameters are all scalars, floats stored as `f32` bit patterns so a
/// [`ComputedConst`] stays `Copy + Eq + Hash` (graph identity folds it) and small (it sits in every
/// value's [`Storage`](crate::Storage)): extents are `u32`, and YaRN's attention factor is resolved
/// when the spec is made. LongRoPE's per-frequency arrays have no scalar form; they stay a
/// [`rope_tables`] input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopeScaling {
    Plain,
    Linear {
        factor_bits: u32,
    },
    DynamicNtk {
        factor_bits: u32,
        original: u32,
    },
    Yarn {
        factor_bits: u32,
        original: u32,
        beta_fast_bits: u32,
        beta_slow_bits: u32,
        attention_factor_bits: u32,
    },
    Llama3 {
        factor_bits: u32,
        low_freq_factor_bits: u32,
        high_freq_factor_bits: u32,
        original: u32,
    },
}

/// The config facts of one RoPE table: rotary width, base and scaling. With a table and a position
/// count it is a [`ComputedConst::Rope`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RopeSpec {
    dim: u32,
    theta_bits: u32,
    scaling: RopeScaling,
}

/// Why a [`RopeSpec`] or its computed constant cannot be made.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RopeSpecError {
    #[error(
        "LongRoPE's per-frequency factors have no scalar form, so a computed RoPE constant cannot hold them"
    )]
    FactorArrays,
    #[error("RoPE {what} {value} does not fit a computed constant's u32 extent")]
    Extent { what: &'static str, value: usize },
}

fn extent(what: &'static str, value: usize) -> Result<u32, RopeSpecError> {
    u32::try_from(value).map_err(|_| RopeSpecError::Extent { what, value })
}

impl RopeSpec {
    /// The spec of `flavor` at rotary width `dim` and base `theta`, every float kept bit for bit.
    pub fn new(dim: usize, theta: f32, flavor: &RopeFlavor<'_>) -> Result<Self, RopeSpecError> {
        let scaling = match *flavor {
            RopeFlavor::Plain => RopeScaling::Plain,
            RopeFlavor::Linear { factor } => RopeScaling::Linear {
                factor_bits: factor.to_bits(),
            },
            RopeFlavor::DynamicNtk { factor, original } => RopeScaling::DynamicNtk {
                factor_bits: factor.to_bits(),
                original: extent("original context", original)?,
            },
            RopeFlavor::Yarn {
                factor,
                original,
                beta_fast,
                beta_slow,
                attention_factor,
            } => RopeScaling::Yarn {
                factor_bits: factor.to_bits(),
                original: extent("original context", original)?,
                beta_fast_bits: beta_fast.to_bits(),
                beta_slow_bits: beta_slow.to_bits(),
                attention_factor_bits: yarn_attention_factor(factor, attention_factor).to_bits(),
            },
            RopeFlavor::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original,
            } => RopeScaling::Llama3 {
                factor_bits: factor.to_bits(),
                low_freq_factor_bits: low_freq_factor.to_bits(),
                high_freq_factor_bits: high_freq_factor.to_bits(),
                original: extent("original context", original)?,
            },
            RopeFlavor::LongRope { .. } => return Err(RopeSpecError::FactorArrays),
        };
        Ok(Self {
            dim: extent("rotary width", dim)?,
            theta_bits: theta.to_bits(),
            scaling,
        })
    }

    /// The rotary width: the table's column count.
    pub fn dim(self) -> usize {
        self.dim as usize
    }

    pub fn theta(self) -> f32 {
        f32::from_bits(self.theta_bits)
    }

    /// The flavor this spec stores. YaRN's attention factor comes back resolved (`Some`), which
    /// builds the same table as the `None` it may have been made from.
    pub fn flavor(self) -> RopeFlavor<'static> {
        match self.scaling {
            RopeScaling::Plain => RopeFlavor::Plain,
            RopeScaling::Linear { factor_bits } => RopeFlavor::Linear {
                factor: f32::from_bits(factor_bits),
            },
            RopeScaling::DynamicNtk {
                factor_bits,
                original,
            } => RopeFlavor::DynamicNtk {
                factor: f32::from_bits(factor_bits),
                original: original as usize,
            },
            RopeScaling::Yarn {
                factor_bits,
                original,
                beta_fast_bits,
                beta_slow_bits,
                attention_factor_bits,
            } => RopeFlavor::Yarn {
                factor: f32::from_bits(factor_bits),
                original: original as usize,
                beta_fast: f32::from_bits(beta_fast_bits),
                beta_slow: f32::from_bits(beta_slow_bits),
                attention_factor: Some(f32::from_bits(attention_factor_bits)),
            },
            RopeScaling::Llama3 {
                factor_bits,
                low_freq_factor_bits,
                high_freq_factor_bits,
                original,
            } => RopeFlavor::Llama3 {
                factor: f32::from_bits(factor_bits),
                low_freq_factor: f32::from_bits(low_freq_factor_bits),
                high_freq_factor: f32::from_bits(high_freq_factor_bits),
                original: original as usize,
            },
        }
    }

    /// The [`ComputedConst`] for `table` over `positions` rows: the entry's capacity, never the
    /// model's maximum context (R-562-2).
    pub fn computed(
        self,
        table: RopeTable,
        positions: usize,
    ) -> Result<ComputedConst, RopeSpecError> {
        Ok(ComputedConst::Rope {
            table,
            positions: extent("positions", positions)?,
            spec: self,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Card 562a SC-002: tables recorded from master's `poot-llm` `rope_tables` (commit 3cf9cdc10, before
    // the move), as f32 bit patterns, four positions each. Comparing against the moved function itself
    // would hold by construction (R-562-2).
    const PLAIN_COS: [u32; 16] = [
        0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f0a5140, 0x3f7ffcb9, 0x3f0a5140,
        0x3f7ffcb9, 0xbed51133, 0x3f7ff2e5, 0xbed51133, 0x3f7ff2e5, 0xbf7d7026, 0x3f7fe283,
        0xbf7d7026, 0x3f7fe283,
    ];
    const PLAIN_SIN: [u32; 16] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f576aa4, 0x3c23d657, 0x3f576aa4,
        0x3c23d657, 0x3f68c7b7, 0x3ca3d43e, 0x3f68c7b7, 0x3ca3d43e, 0x3e1081c3, 0x3cf5b91f,
        0x3e1081c3, 0x3cf5b91f,
    ];
    const PARTIAL_COS: [u32; 16] = [
        0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f0a5140, 0x3f7ffff8, 0x3f0a5140,
        0x3f7ffff8, 0xbed51133, 0x3f7fffde, 0xbed51133, 0x3f7fffde, 0xbf7d7026, 0x3f7fffb5,
        0xbf7d7026, 0x3f7fffb5,
    ];
    const PARTIAL_SIN: [u32; 16] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f576aa4, 0x3a83126e, 0x3f576aa4,
        0x3a83126e, 0x3f68c7b7, 0x3b031269, 0x3f68c7b7, 0x3b031269, 0x3e1081c3, 0x3b449b93,
        0x3e1081c3, 0x3b449b93,
    ];
    const LINEAR_COS: [u32; 16] = [
        0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f60a940, 0x3f7fff2e, 0x3f60a940,
        0x3f7fff2e, 0x3f0a5140, 0x3f7ffcb9, 0x3f0a5140, 0x3f7ffcb9, 0x3d90deaa, 0x3f7ff8a1,
        0x3d90deaa, 0x3f7ff8a1,
    ];
    const LINEAR_SIN: [u32; 16] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3ef57744, 0x3ba3d6dd, 0x3ef57744,
        0x3ba3d6dd, 0x3f576aa4, 0x3c23d657, 0x3f576aa4, 0x3c23d657, 0x3f7f5bd5, 0x3c75c033,
        0x3f7f5bd5, 0x3c75c033,
    ];
    const DYNAMIC_COS: [u32; 16] = [
        0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f0a5140, 0x3f7fffa3, 0x3f0a5140,
        0x3f7fffa3, 0xbed51133, 0x3f7ffe8b, 0xbed51133, 0x3f7ffe8b, 0xbf7d7026, 0x3f7ffcb9,
        0xbf7d7026, 0x3f7ffcb9,
    ];
    const DYNAMIC_SIN: [u32; 16] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f576aa4, 0x3b5a73f3, 0x3f576aa4,
        0x3b5a73f3, 0x3f68c7b7, 0x3bda73a4, 0x3f68c7b7, 0x3bda73a4, 0x3e1081c3, 0x3c23d657,
        0x3e1081c3, 0x3c23d657,
    ];
    const YARN_COS: [u32; 32] = [
        0x3f8bba80, 0x3f8bba80, 0x3f8bba80, 0x3f8bba80, 0x3f8bba80, 0x3f8bba80, 0x3f8bba80,
        0x3f8bba80, 0x3f16fdc4, 0x3f8b07cc, 0x3f8bb9a0, 0x3f8bba7f, 0x3f16fdc4, 0x3f8b07cc,
        0x3f8bb9a0, 0x3f8bba7f, 0xbee8971f, 0x3f88f179, 0x3f8bb6ff, 0x3f8bba7d, 0xbee8971f,
        0x3f88f179, 0x3f8bb6ff, 0x3f8bba7d, 0xbf8a5487, 0x3f857cdd, 0x3f8bb29d, 0x3f8bba79,
        0xbf8a5487, 0x3f857cdd, 0x3f8bb29d, 0x3f8bba79,
    ];
    const YARN_SIN: [u32; 32] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x3f6b27ac, 0x3ddf3176, 0x3bfa6437, 0x39e4ee63, 0x3f6b27ac, 0x3ddf3176,
        0x3bfa6437, 0x39e4ee63, 0x3f7e1c0c, 0x3e5e1403, 0x3c7a62a5, 0x3a64ee62, 0x3f7e1c0c,
        0x3e5e1403, 0x3c7a62a5, 0x3a64ee62, 0x3e1dbf77, 0x3ea52ba0, 0x3cbbc805, 0x3aabb2c8,
        0x3e1dbf77, 0x3ea52ba0, 0x3cbbc805, 0x3aabb2c8,
    ];
    const LLAMA3_COS: [u32; 24] = [
        0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f0a5140,
        0x3f7ff58a, 0x3f7fffff, 0x3f0a5140, 0x3f7ff58a, 0x3f7fffff, 0xbed51133, 0x3f7fd62a,
        0x3f7ffffe, 0xbed51133, 0x3f7fd62a, 0x3f7ffffe, 0xbf7d7026, 0x3f7fa1e2, 0x3f7ffffb,
        0xbf7d7026, 0x3f7fa1e2, 0x3f7ffffb,
    ];
    const LLAMA3_SIN: [u32; 24] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f576aa4,
        0x3c925a8a, 0x398d3169, 0x3f576aa4, 0x3c925a8a, 0x398d3169, 0x3f68c7b7, 0x3d12548f,
        0x3a0d3169, 0x3f68c7b7, 0x3d12548f, 0x3a0d3169, 0x3e1081c3, 0x3d5b6fe4, 0x3a53ca1c,
        0x3e1081c3, 0x3d5b6fe4, 0x3a53ca1c,
    ];
    const LONGROPE_COS: [u32; 16] = [
        0x3fb504f3, 0x3fb504f3, 0x3fb504f3, 0x3fb504f3, 0x3f9edc02, 0x3fb50494, 0x3f9edc02,
        0x3fb50494, 0x3f439c3e, 0x3fb50377, 0x3f439c3e, 0x3fb50377, 0x3dcce076, 0x3fb5019d,
        0x3dcce076, 0x3fb5019d,
    ];
    const LONGROPE_SIN: [u32; 16] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f2d9212, 0x3bb95d00, 0x3f2d9212,
        0x3bb95d00, 0x3f985290, 0x3c395c9f, 0x3f985290, 0x3c395c9f, 0x3fb490dd, 0x3c8b04fe,
        0x3fb490dd, 0x3c8b04fe,
    ];
    const FREQ_FACTORS_COS: [u32; 16] = [
        0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f0a5140, 0x3f7fffcc, 0x3f0a5140,
        0x3f7fffcc, 0xbed51133, 0x3f7fff2e, 0xbed51133, 0x3f7fff2e, 0xbf7d7026, 0x3f7ffe28,
        0xbf7d7026, 0x3f7ffe28,
    ];
    const FREQ_FACTORS_SIN: [u32; 16] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f576aa4, 0x3b23d6ff, 0x3f576aa4,
        0x3b23d6ff, 0x3f68c7b7, 0x3ba3d6dd, 0x3f68c7b7, 0x3ba3d6dd, 0x3e1081c3, 0x3bf5c1f8,
        0x3e1081c3, 0x3bf5c1f8,
    ];

    /// Every element equal bit for bit, and finite (ADR-0101: NaN always fails).
    fn assert_table(label: &str, got: &[f32], want: &[u32]) {
        assert_eq!(got.len(), want.len(), "{label}: element count");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(g.is_finite(), "{label}[{i}] = {g} is not finite");
            assert_eq!(
                g.to_bits(),
                *w,
                "{label}[{i}]: got {g} ({:#010x}), master recorded {} ({w:#010x})",
                g.to_bits(),
                f32::from_bits(*w)
            );
        }
    }

    /// The computed constant's cos and sin tables against the recorded pair, at a capacity of four.
    fn assert_computed(label: &str, spec: RopeSpec, cos: &[u32], sin: &[u32]) {
        for (table, want) in [(RopeTable::Cos, cos), (RopeTable::Sin, sin)] {
            let c = spec.computed(table, 4).unwrap();
            assert_eq!(c.shape(), vec![4, spec.dim()], "{label} {table:?}: shape");
            assert_table(&format!("{label} {table:?}"), &c.values_f32(), want);
        }
    }

    fn spec(dim: usize, theta: f32, flavor: RopeFlavor<'_>) -> RopeSpec {
        RopeSpec::new(dim, theta, &flavor).expect("a scalar flavor has a spec")
    }

    #[test]
    fn computed_plain_rope_equals_master_table() {
        let s = spec(4, 10_000.0, RopeFlavor::Plain);
        assert_computed("plain", s, &PLAIN_COS, &PLAIN_SIN);
    }

    #[test]
    fn computed_partial_rope_equals_master_table() {
        // A partial rotary table is a plain table at the rotary width (here 4 of a wider head).
        let s = spec(4, 1_000_000.0, RopeFlavor::Plain);
        assert_computed("partial", s, &PARTIAL_COS, &PARTIAL_SIN);
    }

    #[test]
    fn computed_linear_rope_equals_master_table() {
        let s = spec(4, 10_000.0, RopeFlavor::Linear { factor: 2.0 });
        assert_computed("linear", s, &LINEAR_COS, &LINEAR_SIN);
    }

    #[test]
    fn computed_dynamic_ntk_rope_equals_master_table() {
        // Four positions past an original context of two: the base grows.
        let flavor = RopeFlavor::DynamicNtk {
            factor: 2.0,
            original: 2,
        };
        assert_computed(
            "dynamic",
            spec(4, 10_000.0, flavor),
            &DYNAMIC_COS,
            &DYNAMIC_SIN,
        );
    }

    #[test]
    fn computed_yarn_rope_equals_master_table() {
        // A fractional factor (2.5) so a rounded payload (3.0) changes every interpolated frequency
        // and the attention factor.
        let flavor = RopeFlavor::Yarn {
            factor: 2.5,
            original: 2048,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: None,
        };
        assert_computed("yarn", spec(8, 10_000.0, flavor), &YARN_COS, &YARN_SIN);
    }

    #[test]
    fn computed_llama3_rope_equals_master_table() {
        // Width 6 puts one frequency in each band: high (kept), mid (blended), low (divided).
        let flavor = RopeFlavor::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original: 256,
        };
        assert_computed(
            "llama3",
            spec(6, 10_000.0, flavor),
            &LLAMA3_COS,
            &LLAMA3_SIN,
        );
    }

    #[test]
    fn longrope_and_gguf_freq_factor_tables_equal_master() {
        // The two array-parameterized inputs have no computed-constant form; the one formula still
        // reproduces master's tables for them.
        let longrope = RopeFlavor::LongRope {
            original: 2,
            max_positions: 4,
            short: Some(&[1.0, 1.25]),
            long: Some(&[2.0, 2.5]),
        };
        let t = rope_tables(4, 4, 10_000.0, &longrope, None);
        assert_table("longrope cos", &t.cos, &LONGROPE_COS);
        assert_table("longrope sin", &t.sin, &LONGROPE_SIN);
        let t = rope_tables(4, 4, 10_000.0, &RopeFlavor::Plain, Some(&[1.0, 4.0]));
        assert_table("freq_factors cos", &t.cos, &FREQ_FACTORS_COS);
        assert_table("freq_factors sin", &t.sin, &FREQ_FACTORS_SIN);
        assert_eq!(
            RopeSpec::new(4, 10_000.0, &longrope),
            Err(RopeSpecError::FactorArrays)
        );
    }

    #[test]
    fn computed_rope_is_sized_by_capacity() {
        // A plain table's rows do not depend on its length: a two-position entry is the first two rows
        // of the four-position recording, and nothing more.
        let s = spec(4, 10_000.0, RopeFlavor::Plain);
        let c = s.computed(RopeTable::Cos, 2).unwrap();
        assert_eq!(c.shape(), vec![2, 4]);
        assert_table("plain cos, capacity 2", &c.values_f32(), &PLAIN_COS[..8]);
    }

    #[test]
    fn rope_spec_round_trips_every_scalar_flavor_bit_for_bit() {
        let flavors = [
            RopeFlavor::Plain,
            RopeFlavor::Linear { factor: 0.1 },
            RopeFlavor::DynamicNtk {
                factor: 1.7,
                original: 9,
            },
            RopeFlavor::Yarn {
                factor: 2.5,
                original: 7,
                beta_fast: 31.5,
                beta_slow: 0.75,
                attention_factor: Some(1.0e-3),
            },
            RopeFlavor::Llama3 {
                factor: 8.5,
                low_freq_factor: 1.25,
                high_freq_factor: 3.5,
                original: 11,
            },
        ];
        for flavor in flavors {
            let s = spec(6, 123_456.79, flavor);
            assert_eq!(s.flavor(), flavor);
            assert_eq!(s.theta().to_bits(), 123_456.79f32.to_bits());
        }
    }

    #[test]
    fn yarn_default_attention_factor_resolves_to_the_same_table() {
        let unset = RopeFlavor::Yarn {
            factor: 2.5,
            original: 2048,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: None,
        };
        let resolved = spec(8, 10_000.0, unset).flavor();
        assert_eq!(
            resolved,
            RopeFlavor::Yarn {
                factor: 2.5,
                original: 2048,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: Some(0.1 * 2.5f32.ln() + 1.0),
            }
        );
        assert_eq!(
            rope_tables(8, 4, 10_000.0, &resolved, None),
            rope_tables(8, 4, 10_000.0, &unset, None)
        );
    }

    #[test]
    fn extents_past_u32_are_a_typed_refusal() {
        let s = spec(4, 10_000.0, RopeFlavor::Plain);
        let too_many = u32::MAX as usize + 1;
        assert_eq!(
            s.computed(RopeTable::Cos, too_many),
            Err(RopeSpecError::Extent {
                what: "positions",
                value: too_many
            })
        );
        assert!(matches!(
            RopeSpec::new(too_many, 1.0, &RopeFlavor::Plain),
            Err(RopeSpecError::Extent { .. })
        ));
    }

    // The scaled flavors' helpers, against independent references (moved with the formula from
    // `poot-llm/src/checkpoint/gguf/tests/rope_tests.rs`).

    const LLAMA3: RopeFlavor<'static> = RopeFlavor::Llama3 {
        factor: 32.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original: 8192,
    };

    #[test]
    fn llama3_rescale_passes_other_flavors_through() {
        assert_eq!(llama3_rescale(0.5, &RopeFlavor::Plain), 0.5);
        assert_eq!(
            llama3_rescale(0.5, &RopeFlavor::Linear { factor: 32.0 }),
            0.5
        );
    }

    #[test]
    fn llama3_bands() {
        use std::f32::consts::PI;
        // high frequency (short wavelength < old_ctx/high_freq_factor = 2048): untouched.
        let hi = 2.0 * PI / 1000.0;
        assert_eq!(llama3_rescale(hi, &LLAMA3), hi);
        // low frequency (long wavelength > old_ctx/low_freq_factor = 8192): divided by factor.
        let lo = 2.0 * PI / 20000.0;
        assert!((llama3_rescale(lo, &LLAMA3) - lo / 32.0).abs() < 1e-12);
        // mid band: strictly between the scaled and unscaled values.
        let mid = 2.0 * PI / 4096.0;
        let r = llama3_rescale(mid, &LLAMA3);
        assert!(
            r > mid / 32.0 && r < mid,
            "mid-band blend in (f/32, f): {r}"
        );
    }

    #[test]
    fn linear_scaling_divides_inv_freq_by_factor() {
        // HF divides the position by `factor`; dividing inv_freq instead is algebraically identical.
        let s = RopeFlavor::Linear { factor: 4.0 };
        assert!((linear_rescale(2.0, &s) - 0.5).abs() < 1e-7);
        assert!((linear_rescale(1.0, &s) - 0.25).abs() < 1e-7);
        assert_eq!(
            linear_rescale(0.5, &RopeFlavor::Linear { factor: 1.0 }),
            0.5
        );
    }

    #[test]
    fn linear_scaling_passes_other_flavors_and_nonpositive_factors_through() {
        assert_eq!(linear_rescale(0.5, &RopeFlavor::Plain), 0.5);
        assert_eq!(linear_rescale(0.5, &LLAMA3), 0.5);
        assert_eq!(
            linear_rescale(0.5, &RopeFlavor::Linear { factor: 0.0 }),
            0.5
        );
    }

    fn dynamic(factor: f32, original: usize) -> RopeFlavor<'static> {
        RopeFlavor::DynamicNtk { factor, original }
    }

    #[test]
    fn dynamic_ntk_passes_other_flavors_and_short_contexts_through() {
        assert_eq!(
            dynamic_ntk_base(10000.0, &RopeFlavor::Plain, 100_000, 64),
            10000.0
        );
        assert_eq!(dynamic_ntk_base(10000.0, &LLAMA3, 100_000, 64), 10000.0);
        let s = dynamic(4.0, 8192);
        assert_eq!(dynamic_ntk_base(10000.0, &s, 8192, 64), 10000.0);
        assert_eq!(dynamic_ntk_base(10000.0, &s, 100, 64), 10000.0);
    }

    #[test]
    fn dynamic_ntk_matches_hf_reference_formula_beyond_original_context() {
        // HF `_compute_dynamic_ntk_parameters`, reimplemented independently in f64:
        //   base' = base * ((factor * seq_len / orig) - (factor - 1)) ** (dim / (dim - 2))
        let (theta, factor, orig, dim, seq_len) =
            (10000.0f32, 4.0f32, 8192usize, 64usize, 32768usize);
        let got = dynamic_ntk_base(theta, &dynamic(factor, orig), seq_len, dim);
        let want = (theta as f64)
            * ((factor as f64 * seq_len as f64 / orig as f64) - (factor as f64 - 1.0))
                .powf(dim as f64 / (dim as f64 - 2.0));
        assert!(
            ((got as f64 - want) / want).abs() < 1e-6,
            "got={got} want={want}"
        );
        assert!(got > theta, "base must grow beyond the original context");
    }

    fn yarn(theta: f32, d: usize, factor: f32, orig: usize, af: Option<f32>) -> (Vec<f32>, f32) {
        yarn_inv_freq(theta, d, factor, orig, 32.0, 1.0, af)
    }

    #[test]
    fn yarn_attention_factor_matches_default_get_mscale_unless_explicit() {
        // HF default `get_mscale`: 1.0 when factor <= 1, else 0.1*ln(factor) + 1.0.
        let (_, af) = yarn(10000.0, 64, 1.0, 8192, None);
        assert!((af - 1.0).abs() < 1e-7);
        let (_, af) = yarn(10000.0, 64, 8.0, 8192, None);
        let want = 0.1 * 8.0f32.ln() + 1.0;
        assert!((af - want).abs() < 1e-6, "af={af} want={want}");
        let (_, af) = yarn(10000.0, 64, 8.0, 8192, Some(2.5));
        assert_eq!(af, 2.5);
    }

    #[test]
    fn yarn_degenerates_to_plain_rope_at_factor_one() {
        // At factor=1, interpolation == extrapolation for every j, so inv_freq is the unscaled base.
        let (theta, d) = (10000.0f32, 64usize);
        let (inv_freq, af) = yarn(theta, d, 1.0, 8192, None);
        assert!((af - 1.0).abs() < 1e-7);
        for (j, &f) in inv_freq.iter().enumerate() {
            let plain = theta.powf(-((2 * j) as f32) / d as f32);
            assert!(
                (f - plain).abs() < 1e-6,
                "j={j}: yarn(factor=1)={f} plain={plain}"
            );
        }
    }

    #[test]
    fn yarn_matches_hf_reference_formula() {
        // Independent f64 reimplementation of HF `_compute_yarn_parameters`'s NTK-by-parts ramp.
        let (theta, d, factor, orig) = (10000.0f32, 64usize, 8.0f32, 8192usize);
        let (got, _) = yarn(theta, d, factor, orig, None);
        let (theta64, factor64, orig64) = (theta as f64, factor as f64, orig as f64);
        let correction_dim = |num_rotations: f64| -> f64 {
            (d as f64 * (orig64 / (num_rotations * 2.0 * std::f64::consts::PI)).ln())
                / (2.0 * theta64.ln())
        };
        let low = correction_dim(32.0).floor().max(0.0);
        let high = correction_dim(1.0).ceil().min((d - 1) as f64);
        let (low, high) = if low == high {
            (low, high + 0.001)
        } else {
            (low, high)
        };
        for (j, &g) in got.iter().enumerate().take(d / 2) {
            let pos_freq = theta64.powf((2 * j) as f64 / d as f64);
            let extrap = 1.0 / pos_freq;
            let interp = 1.0 / (factor64 * pos_freq);
            let ramp = ((j as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let extrap_factor = 1.0 - ramp;
            let want = interp * (1.0 - extrap_factor) + extrap * extrap_factor;
            assert!((g as f64 - want).abs() < 1e-4, "j={j}: got={g} want={want}");
        }
    }

    #[test]
    fn yarn_short_wavelength_dims_extrapolate_long_wavelength_dims_interpolate() {
        // NTK-by-parts must not rescale every dimension the same way (that would be linear scaling).
        let (theta, d, factor, orig) = (10000.0f32, 64usize, 8.0f32, 8192usize);
        let (inv_freq, _) = yarn(theta, d, factor, orig, None);
        let half = d / 2;
        assert!(
            (inv_freq[0] - 1.0).abs() < 1e-3,
            "j=0 should be (near) fully extrapolated: {}",
            inv_freq[0]
        );
        let interp_last = 1.0 / (factor * theta.powf((2 * (half - 1)) as f32 / d as f32));
        assert!(
            (inv_freq[half - 1] - interp_last).abs() < 1e-3,
            "last dim should be (near) fully interpolated: {} vs {interp_last}",
            inv_freq[half - 1]
        );
        assert!(
            inv_freq[0] > inv_freq[half - 1],
            "inv_freq must still decrease with j under YaRN, same as plain RoPE"
        );
    }

    fn longrope(max_positions: usize, long: Option<&'static [f32]>) -> RopeFlavor<'static> {
        RopeFlavor::LongRope {
            original: 8192,
            max_positions,
            short: Some(&[1.0, 1.0]),
            long,
        }
    }

    /// Column 1's angle at position 1 is that frequency's inverse frequency, so it shows which
    /// factor array the table used.
    fn longrope_inv_freq_1(positions: usize, long: Option<&'static [f32]>) -> f32 {
        let t = rope_tables(4, positions, 10_000.0, &longrope(positions, long), None);
        let af = t.cos[0];
        (t.sin[4 + 1] / af).asin()
    }

    #[test]
    fn longrope_selects_short_factors_at_or_below_the_original_context() {
        // HF's `seq_len <= original_max_position_embeddings` guard, with the table's positions as the
        // sequence length.
        let plain = 10_000.0f32.powf(-0.5);
        for positions in [2, 8192] {
            let got = longrope_inv_freq_1(positions, Some(&[2.0, 2.0]));
            assert!((got - plain).abs() < 1e-6, "{positions}: {got} vs {plain}");
        }
    }

    #[test]
    fn longrope_selects_long_factors_beyond_the_original_context_and_falls_back_to_short() {
        let plain = 10_000.0f32.powf(-0.5);
        let got = longrope_inv_freq_1(8193, Some(&[2.0, 2.0]));
        assert!(
            (got - plain / 2.0).abs() < 1e-6,
            "long: {got} vs {}",
            plain / 2.0
        );
        let got = longrope_inv_freq_1(8193, None);
        assert!(
            (got - plain).abs() < 1e-6,
            "short fallback: {got} vs {plain}"
        );
    }
}
