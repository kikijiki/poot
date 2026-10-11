//! Half-split rotary position embedding at absolute positions, its checked parameters, and the HF
//! `rope_scaling` block as a [`RopeFlavor`].
//!
//! The cos/sin tables are [`ComputedConst::Rope`](poot_graph_ir::ComputedConst::Rope) constants
//! sized by the step's capacity, not weights; a step gathers its rows by the
//! absolute positions in `Slot::Pos`, so a chunk continuing a sequence rotates at its true
//! positions.

use poot_graph_ir::rope_table::{RopeFlavor, RopeSpec, RopeSpecError, RopeTable, rope_tables};
use poot_graph_ir::{BinOp, Builder, Traced, UnOp, ops};

use poot_quant::weights::WeightEntry;
use poot_tensor::DType;

use super::standard::ParamError;
use crate::model::{ConfigReason, FamilyKey, ModelError};

/// Which dims a rotation pairs: the two halves of the rotary width (HF checkpoints), or adjacent
/// even/odd dims (llama.cpp's GGUF Q/K rows, stored permuted for that layout). Both rotate the same
/// pairs with the same frequencies; they differ in where the pair sits in the stored row order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RopePairing {
    HalfSplit,
    Interleaved,
}

/// A checked RoPE: rotary width (even, non-zero), base and scaling flavor, and dim pairing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeParams {
    spec: RopeSpec,
    pairing: RopePairing,
    sections: Option<[usize; 3]>,
}

fn positive(field: &'static str, value: f32) -> Result<f32, ParamError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(ParamError::new(field, ConfigReason::NotFinitePositive))
    }
}

fn nonzero(field: &'static str, value: usize) -> Result<usize, ParamError> {
    if value == 0 {
        Err(ParamError::new(field, ConfigReason::Zero))
    } else {
        Ok(value)
    }
}

impl RopeParams {
    pub fn new(rotary_dim: usize, theta: f32, flavor: &RopeFlavor<'_>) -> Result<Self, ParamError> {
        nonzero("rotary_dim", rotary_dim)?;
        if !rotary_dim.is_multiple_of(2) {
            return Err(ParamError::new(
                "rotary_dim",
                ConfigReason::NotDivisible { by: 2 },
            ));
        }
        positive("rope_theta", theta)?;
        match *flavor {
            RopeFlavor::Plain | RopeFlavor::LongRope { .. } => {}
            RopeFlavor::Proportional { rotated } => {
                if rotated > rotary_dim / 2 {
                    return Err(ParamError::new(
                        "rotated_pairs",
                        ConfigReason::Exceeds {
                            max: rotary_dim / 2,
                        },
                    ));
                }
            }
            RopeFlavor::Linear { factor } => {
                positive("rope_scaling.factor", factor)?;
            }
            RopeFlavor::DynamicNtk { factor, original }
            | RopeFlavor::Yarn {
                factor, original, ..
            }
            | RopeFlavor::Llama3 {
                factor, original, ..
            } => {
                positive("rope_scaling.factor", factor)?;
                nonzero("rope_scaling.original_max_position_embeddings", original)?;
            }
        }
        let spec = RopeSpec::new(rotary_dim, theta, flavor).map_err(|e| match e {
            RopeSpecError::FactorArrays => {
                ParamError::new("rope_scaling", ConfigReason::Unsupported)
            }
            RopeSpecError::Extent { what, .. } => ParamError::new(
                what,
                ConfigReason::Exceeds {
                    max: u32::MAX as usize,
                },
            ),
        })?;
        Ok(Self {
            spec,
            pairing: RopePairing::HalfSplit,
            sections: None,
        })
    }

    /// Sectioned ("mRoPE") rotation: the rotary pairs split into temporal, height and width groups
    /// of `sections` pairs each, every group rotating at its own axis of `Slot::MropePosition`
    /// ([`rope_sectioned_rows`]). The widths sum to half the rotary width; an interleaved pairing has
    /// no sectioned form.
    pub fn sectioned(self, sections: [usize; 3]) -> Result<Self, ParamError> {
        const FIELD: &str = "rope_scaling.mrope_section";
        if self.pairing != RopePairing::HalfSplit {
            return Err(ParamError::new(FIELD, ConfigReason::Unsupported));
        }
        let pairs = sections
            .iter()
            .try_fold(0usize, |sum, &width| sum.checked_add(width))
            .ok_or(ParamError::new(FIELD, ConfigReason::Overflow))?;
        if pairs != self.rotary() / 2 {
            return Err(ParamError::new(
                FIELD,
                ConfigReason::Mismatch {
                    expected: self.rotary() / 2,
                    found: pairs,
                },
            ));
        }
        Ok(Self {
            sections: Some(sections),
            ..self
        })
    }

    /// The section widths of a sectioned rotation, in rotary pairs.
    pub fn sections(self) -> Option<[usize; 3]> {
        self.sections
    }

    /// The same rotation over adjacent even/odd dim pairs ([`RopePairing::Interleaved`]).
    pub fn interleaved(self) -> Self {
        Self {
            pairing: RopePairing::Interleaved,
            ..self
        }
    }

    /// The rotary width: the leading head dims that rotate.
    pub fn rotary(self) -> usize {
        self.spec.dim()
    }

    /// The same rotary width, base and pairing rescaled by llama.cpp's per-frequency `factors`
    /// (`rope_freqs.weight`), see [`llama3_flavor_from_factors`].
    pub fn with_freq_factors(self, factors: &[f32]) -> Result<Self, ParamError> {
        let field = "rope_freqs.weight";
        let flavor = llama3_flavor_from_factors(factors, self.rotary(), self.spec.theta())
            .map_err(|reason| ParamError::new(field, reason))?;
        Ok(Self {
            pairing: self.pairing,
            sections: self.sections,
            ..Self::new(self.rotary(), self.spec.theta(), &flavor)?
        })
    }
}

/// The GGUF tensor llama.cpp writes for a scaled or partial RoPE: the per-frequency factors the
/// config's RoPE is rescaled by.
pub const GGUF_ROPE_FREQS: &str = "rope_freqs.weight";

/// The per-frequency factors of a `rope_freqs.weight`: llama.cpp writes them as one F32 vector.
pub fn rope_freq_factors(family: FamilyKey, entry: &WeightEntry) -> Result<Vec<f32>, ModelError> {
    match entry {
        WeightEntry::Dense(dense) if dense.dtype() == DType::F32 => Ok(dense
            .bytes()
            .as_slice()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()),
        _ => Err(ModelError::Config {
            family,
            field: GGUF_ROPE_FREQS,
            reason: ConfigReason::WrongType,
        }),
    }
}

/// How far a decoded llama3 rescale may sit from the stored factors, as the largest absolute
/// cos/sin difference over the checked positions. The stored factors are `f32`, so a faithful
/// decode lands orders of magnitude below it; a tensor that is not a llama3 rescale does not.
const LLAMA3_DECODE_TOLERANCE: f32 = 1e-3;

/// Positions over which a decoded rescale is checked against the stored factors: past llama3's
/// original context for every checkpoint llama.cpp converts, so the low-frequency band, where the
/// factors matter, rotates far enough to show an error.
const LLAMA3_DECODE_POSITIONS: usize = 8192;

/// A factor at least this large divides a frequency to nothing: llama.cpp's Gemma 4 converter
/// (`Gemma4Model.generate_extra_tensors`) writes `1e30` for a pair that does not rotate, as the
/// real 31B file's `rope_freqs.weight` shows (64 ones, then `1e30`).
const NOPE_FACTOR: f32 = 1e20;

/// The number of rotated pairs of a proportional `rope_freqs.weight` (Gemma 4): ones up to a point,
/// then only NoPE factors. All-ones and any other shape are not proportional.
fn proportional_pairs(factors: &[f32]) -> Option<usize> {
    let rotated = factors.iter().take_while(|f| **f == 1.0).count();
    let tail = &factors[rotated..];
    (!tail.is_empty() && tail.iter().all(|f| *f >= NOPE_FACTOR)).then_some(rotated)
}

/// The llama3 flavor a llama.cpp `rope_freqs.weight` was written from.
///
/// llama.cpp folds llama3's per-frequency rescale into that tensor (`inv_freq[j] / factors[j]`) and
/// carries none of the scaling fields in metadata, while a table's [`RopeFlavor`] holds scalars
/// only. The tensor is a pure function of `(factor, low_freq_factor, high_freq_factor, original)`
/// over the rotary width and base, so this decodes those back: `factor` is the plateau the
/// low-frequency band is divided by, and the blended band between the two wavelength cutoffs pins
/// the affine `smooth = a / wavelength - b` (with `a = original / (high - low)` and
/// `b = low / (high - low)`) that the HF formula applies there. Only that ratio is observable, so the
/// decode takes `original` as the integer nearest `a / b` and derives the rest. That canonicalizes the
/// config `(original, low, high)` to the equivalent flavor with `low = 1`: the wavelength cutoffs
/// `original / low` and `original / high` are identical, so the tables are too (Llama 3.1 and 3.2
/// both have `low = 1`).
///
/// A genuine llama3 rescale with fewer than two blended frequencies (tiny heads or extreme
/// parameters) pins no slope and is refused as [`ConfigReason::Unsupported`].
///
/// The decode is checked: the tables it builds must equal the tables the stored factors build, so a
/// tensor that is not a llama3 rescale is [`ConfigReason::Unsupported`], never silently approximated.
/// All-ones factors are plain RoPE.
fn llama3_flavor_from_factors(
    factors: &[f32],
    rotary: usize,
    theta: f32,
) -> Result<RopeFlavor<'static>, ConfigReason> {
    if factors.len() != rotary / 2 {
        return Err(ConfigReason::WrongType);
    }
    if factors.iter().any(|f| !(f.is_finite() && *f >= 1.0)) {
        return Err(ConfigReason::Unsupported);
    }
    if let Some(rotated) = proportional_pairs(factors) {
        return Ok(RopeFlavor::Proportional { rotated });
    }
    let plateau = factors.iter().copied().fold(1.0f32, f32::max);
    if plateau == 1.0 {
        return Ok(RopeFlavor::Plain);
    }
    // 1 / wavelength of frequency j, and the blend weight its factor implies.
    let inverse_wavelength =
        |j: usize| theta.powf(-((2 * j) as f32) / rotary as f32) / (2.0 * std::f32::consts::PI);
    let smooth = |f: f32| (f.recip() - plateau.recip()) / (1.0 - plateau.recip());
    let edge = 1e-4 * plateau;
    let mut blended =
        (0..factors.len()).filter(|&j| factors[j] > 1.0 + edge && factors[j] < plateau - edge);
    let (first, last) = match (blended.next(), blended.next_back()) {
        (Some(first), Some(last)) => (first, last),
        _ => return Err(ConfigReason::Unsupported),
    };
    let slope = (smooth(factors[last]) - smooth(factors[first]))
        / (inverse_wavelength(last) - inverse_wavelength(first));
    let offset = slope * inverse_wavelength(first) - smooth(factors[first]);
    if !(slope.is_finite() && slope > 0.0 && offset.is_finite() && offset > 0.0) {
        return Err(ConfigReason::Unsupported);
    }
    let original = (slope / offset).round();
    if !(original >= 1.0 && original <= u32::MAX as f32) {
        return Err(ConfigReason::Unsupported);
    }
    let low_freq_factor = original * offset / slope;
    let flavor = RopeFlavor::Llama3 {
        factor: plateau,
        low_freq_factor,
        high_freq_factor: low_freq_factor + original / slope,
        original: original as usize,
    };
    let stored = rope_tables(
        rotary,
        LLAMA3_DECODE_POSITIONS,
        theta,
        &RopeFlavor::Plain,
        Some(factors),
    );
    let decoded = rope_tables(rotary, LLAMA3_DECODE_POSITIONS, theta, &flavor, None);
    let worst = stored
        .cos
        .iter()
        .zip(&decoded.cos)
        .chain(stored.sin.iter().zip(&decoded.sin))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if worst > LLAMA3_DECODE_TOLERANCE {
        return Err(ConfigReason::Unsupported);
    }
    Ok(flavor)
}

/// The flavor an HF `rope_scaling` block names (`rope_type`, or Phi's `type`): absent, `null` or
/// `default` is plain; `linear`, `dynamic`, `yarn` and `llama3` read their HF fields (YaRN's
/// `beta_fast`/`beta_slow` default to HF's 32 and 1; dynamic NTK's and an omitted YaRN original
/// context are `max_positions`, as HF reads them). Any other type, including `longrope` (whose
/// per-frequency arrays a computed table cannot hold), is a typed refusal.
pub fn hf_rope_flavor(
    scaling: Option<&serde_json::Value>,
    max_positions: usize,
) -> Result<RopeFlavor<'static>, ParamError> {
    let wrong = |field| ParamError::new(field, ConfigReason::WrongType);
    let missing = |field| ParamError::new(field, ConfigReason::Missing);
    let scaling = match scaling {
        None | Some(serde_json::Value::Null) => return Ok(RopeFlavor::Plain),
        Some(serde_json::Value::Object(scaling)) => scaling,
        Some(_) => return Err(wrong("rope_scaling")),
    };
    let kind = match scaling.get("rope_type").or_else(|| scaling.get("type")) {
        None => return Err(missing("rope_scaling.rope_type")),
        Some(kind) => kind.as_str().ok_or(wrong("rope_scaling.rope_type"))?,
    };
    let float = |field: &'static str, key: &str| -> Result<Option<f32>, ParamError> {
        scaling
            .get(key)
            .filter(|v| !v.is_null())
            .map(|v| v.as_f64().map(|v| v as f32).ok_or(wrong(field)))
            .transpose()
    };
    let count = |field: &'static str, key: &str| -> Result<Option<usize>, ParamError> {
        scaling
            .get(key)
            .filter(|v| !v.is_null())
            .map(|v| v.as_u64().map(|v| v as usize).ok_or(wrong(field)))
            .transpose()
    };
    let factor = || float("rope_scaling.factor", "factor")?.ok_or(missing("rope_scaling.factor"));
    let original = || {
        count(
            "rope_scaling.original_max_position_embeddings",
            "original_max_position_embeddings",
        )
    };
    Ok(match kind {
        "default" => RopeFlavor::Plain,
        "linear" => RopeFlavor::Linear { factor: factor()? },
        "dynamic" => RopeFlavor::DynamicNtk {
            factor: factor()?,
            original: max_positions,
        },
        "yarn" => {
            if scaling.contains_key("mscale") || scaling.contains_key("mscale_all_dim") {
                return Err(ParamError::new(
                    "rope_scaling.mscale",
                    ConfigReason::Unsupported,
                ));
            }
            RopeFlavor::Yarn {
                factor: factor()?,
                original: original()?.unwrap_or(max_positions),
                beta_fast: float("rope_scaling.beta_fast", "beta_fast")?.unwrap_or(32.0),
                beta_slow: float("rope_scaling.beta_slow", "beta_slow")?.unwrap_or(1.0),
                attention_factor: float("rope_scaling.attention_factor", "attention_factor")?,
            }
        }
        "llama3" => RopeFlavor::Llama3 {
            factor: factor()?,
            low_freq_factor: float("rope_scaling.low_freq_factor", "low_freq_factor")?
                .ok_or(missing("rope_scaling.low_freq_factor"))?,
            high_freq_factor: float("rope_scaling.high_freq_factor", "high_freq_factor")?
                .ok_or(missing("rope_scaling.high_freq_factor"))?,
            original: original()?
                .ok_or(missing("rope_scaling.original_max_position_embeddings"))?,
        },
        _ => {
            return Err(ParamError::new(
                "rope_scaling.rope_type",
                ConfigReason::Unsupported,
            ));
        }
    })
}

/// The cos and sin rows of one step's positions, `[rows, 1, tokens, rotary]` (broadcast over
/// heads).
#[derive(Clone, Copy, Debug)]
pub struct RopeRows {
    cos: Traced,
    sin: Traced,
    rotary: usize,
    pairing: RopePairing,
}

/// Gather the step's rows of `p`'s tables over `capacity` positions at `pos` (`[rows, tokens]`
/// I32 absolute positions, each below `capacity`).
pub fn rope_rows(b: &Builder, p: RopeParams, pos: Traced, capacity: usize) -> RopeRows {
    let shape = b.aval(pos).shape;
    let (rows, tokens) = (shape[0], shape[1]);
    let rotary = p.rotary();
    let table = |table| {
        let computed = p
            .spec
            .computed(table, capacity)
            .expect("a step's capacity fits a u32: StepShape capacities are checked by the model");
        let full = b.computed(computed);
        b.reshape(b.gather(full, 0, pos), vec![rows, 1, tokens, rotary])
    };
    RopeRows {
        cos: table(RopeTable::Cos),
        sin: table(RopeTable::Sin),
        rotary,
        pairing: p.pairing,
    }
}

/// Rotate `x` (`[rows, heads, tokens, head_dim]`) by `p`'s sectioned rotation at the three axes of
/// `mrope` (`[3, rows * tokens]` I32, axis-major temporal, height, width): `ops::rope_sectioned`
/// over the tables of `p`, each section's pairs taking their cos/sin from their own axis. Equal
/// axes give the plain rotation at that position. `p` must be [`RopeParams::sectioned`].
pub fn apply_rope_sectioned(
    b: &Builder,
    p: RopeParams,
    x: Traced,
    mrope: Traced,
    capacity: usize,
) -> Traced {
    let sections = p
        .sections
        .expect("apply_rope_sectioned needs a sectioned RopeParams");
    let shape = b.aval(x).shape;
    let (rows, tokens) = (shape[0], shape[2]);
    let table = |table| {
        let computed = p
            .spec
            .computed(table, capacity)
            .expect("a step's capacity fits a u32: StepShape capacities are checked by the model");
        b.computed(computed)
    };
    let axes: Vec<Traced> = (0..3)
        .map(|axis| b.reshape(b.slice(mrope, 0, axis, axis + 1), vec![rows, 1, tokens]))
        .collect();
    ops::rope_sectioned(
        b,
        x,
        table(RopeTable::Cos),
        table(RopeTable::Sin),
        &axes,
        &sections,
    )
}

/// The rows of `p`'s tables at positions `0, stride, 2 * stride, ..`, `count` of them, for `rows`
/// sequences: `[rows, 1, count, rotary]`, rows of compressed windows that start at every `stride`th
/// position (known at trace time, so a view of the table and no gather). `count * stride` must not
/// exceed `capacity`.
pub fn rope_strided_rows(
    b: &Builder,
    p: RopeParams,
    capacity: usize,
    stride: usize,
    count: usize,
    rows: usize,
) -> RopeRows {
    let rotary = p.rotary();
    let table = |table| {
        let computed = p
            .spec
            .computed(table, capacity)
            .expect("a step's capacity fits a u32: StepShape capacities are checked by the model");
        let full = b.computed(computed);
        let tiles = b.reshape(
            b.slice(full, 0, 0, count * stride),
            vec![count, stride, rotary],
        );
        let starts = b.reshape(b.slice(tiles, 1, 0, 1), vec![1, 1, count, rotary]);
        b.broadcast(starts, vec![rows, 1, count, rotary])
    };
    RopeRows {
        cos: table(RopeTable::Cos),
        sin: table(RopeTable::Sin),
        rotary,
        pairing: p.pairing,
    }
}

impl RopeRows {
    /// The inverse rotation: the same rows with `sin` negated, so [`apply_rope`] undoes the
    /// rotation by the same positions (DeepSeek-V4 derotates its attention output, which picked up
    /// the rotation of the shared key it reads as a value).
    pub fn inverse(self, b: &Builder) -> Self {
        Self {
            sin: b.unary(UnOp::Neg, self.sin),
            ..self
        }
    }
}

/// Rotate the leading `rotary` dims of `x` (`[rows, heads, tokens, head_dim]`): half-split,
/// `x * cos + concat(-x2, x1) * sin`, or interleaved, each adjacent pair `(a, b)` to
/// `(a cos - b sin, b cos + a sin)`; the trailing dims pass through.
pub fn apply_rope(b: &Builder, x: Traced, r: &RopeRows) -> Traced {
    let d = *b.aval(x).shape.last().expect("rope input has a head axis");
    let (rot, half) = (r.rotary, r.rotary / 2);
    let axis = 3;
    let part = if rot == d {
        x
    } else {
        b.slice(x, axis, 0, rot)
    };
    let out = match r.pairing {
        RopePairing::HalfSplit => {
            let x1 = b.slice(part, axis, 0, half);
            let x2 = b.slice(part, axis, half, rot);
            let neg_x2 = b.unary(UnOp::Neg, x2);
            let rotated = b.concat(axis, &[neg_x2, x1]);
            b.binary(
                BinOp::Add,
                b.binary(BinOp::Mul, part, r.cos),
                b.binary(BinOp::Mul, rotated, r.sin),
            )
        }
        RopePairing::Interleaved => {
            let shape = b.aval(part).shape;
            let pairs = vec![shape[0], shape[1], shape[2], half, 2];
            let part = b.reshape(part, pairs);
            let lane = |i| {
                let v = b.slice(part, 4, i, i + 1);
                b.reshape(v, vec![shape[0], shape[1], shape[2], half])
            };
            let (even, odd) = (lane(0), lane(1));
            // The tables repeat each frequency in both halves; the first half is one per pair.
            let (cos, sin) = (b.slice(r.cos, axis, 0, half), b.slice(r.sin, axis, 0, half));
            let mul = |a, c| b.binary(BinOp::Mul, a, c);
            let new_even = b.binary(BinOp::Sub, mul(even, cos), mul(odd, sin));
            let new_odd = b.binary(BinOp::Add, mul(odd, cos), mul(even, sin));
            let pair = |v| b.reshape(v, vec![shape[0], shape[1], shape[2], half, 1]);
            let joined = b.concat(4, &[pair(new_even), pair(new_odd)]);
            b.reshape(joined, vec![shape[0], shape[1], shape[2], rot])
        }
    };
    if rot == d {
        out
    } else {
        b.concat(axis, &[out, b.slice(x, axis, rot, d)])
    }
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{Slot, TensorType};
    use poot_tensor::DType;
    use serde_json::json;

    use super::*;
    use crate::components::standard::oracle::{self, Fx};

    /// SC-001: plain RoPE at arbitrary absolute positions (a continuation, not `0..t`) against an
    /// f64 half-split rotation with `theta^(-2j/rotary)` frequencies, full and partial rotary.
    #[test]
    fn apply_rope_matches_an_f64_rotation_at_absolute_positions() {
        let (heads, t, d, cap) = (2, 3, 8, 16);
        let theta = 10_000.0f32;
        let pos = [5, 6, 11];
        let x = oracle::random(heads * t * d, 21);
        for rotary in [d, 4] {
            let p = RopeParams::new(rotary, theta, &RopeFlavor::Plain).unwrap();
            let b = Builder::new();
            let ps = b.slot(Slot::Pos, TensorType::new(vec![1, t], DType::I32));
            let xs = b.constant("x", TensorType::f32(vec![1, heads, t, d]));
            let rows = rope_rows(&b, p, ps, cap);
            let y = apply_rope(&b, xs, &rows);
            let got = Fx::new(0)
                .eval(
                    &b.finish(y).unwrap(),
                    &[("x", &x)],
                    &[(Slot::Pos, &pos)],
                    &[],
                )
                .0;
            let half = rotary / 2;
            let mut want: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
            for h in 0..heads {
                for (i, &p) in pos.iter().enumerate() {
                    let base = (h * t + i) * d;
                    for j in 0..half {
                        let inv = f64::from(theta).powf(-((2 * j) as f64) / rotary as f64);
                        let (s, c) = (f64::from(p) * inv).sin_cos();
                        let (a, bb) = (f64::from(x[base + j]), f64::from(x[base + half + j]));
                        want[base + j] = a * c - bb * s;
                        want[base + half + j] = bb * c + a * s;
                    }
                }
            }
            oracle::assert_matches_f64(&got, &want, 1e-5);
        }
    }

    /// Sectioned RoPE rotates each rotary pair at its own axis: pairs `0..1` by the temporal id, `1..3`
    /// by the height id and `3..4` by the width id (sections `[1, 2, 1]` over 4 pairs), against an
    /// f64 reference whose axes differ at every token. Mutation: swap the first two section widths
    /// in `apply_rope_sectioned`; the pairs change axis and this goes red.
    #[test]
    fn sectioned_rope_rotates_each_pair_at_its_sections_axis() {
        let (heads, t, d, cap) = (2, 3, 8, 16);
        let theta = 10_000.0f32;
        let sections = [1usize, 2, 1];
        let axes: [[i32; 3]; 3] = [[5, 6, 11], [2, 3, 3], [7, 9, 4]];
        let packed: Vec<i32> = axes.iter().flatten().copied().collect();
        let x = oracle::random(heads * t * d, 23);
        let p = RopeParams::new(d, theta, &RopeFlavor::Plain)
            .unwrap()
            .sectioned(sections)
            .unwrap();
        let b = Builder::new();
        let ms = b.slot(Slot::MropePosition, TensorType::new(vec![3, t], DType::I32));
        let xs = b.constant("x", TensorType::f32(vec![1, heads, t, d]));
        let y = apply_rope_sectioned(&b, p, xs, ms, cap);
        let got = Fx::new(0)
            .eval(
                &b.finish(y).unwrap(),
                &[("x", &x)],
                &[(Slot::MropePosition, &packed)],
                &[],
            )
            .0;
        let axis_of_pair = |j: usize| match j {
            0 => 0,
            1..3 => 1,
            _ => 2,
        };
        let position_of = |pair: usize, token: usize| axes[axis_of_pair(pair)][token];
        let half = d / 2;
        let mut want: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        for h in 0..heads {
            for i in 0..t {
                let base = (h * t + i) * d;
                for j in 0..half {
                    let position = position_of(j, i);
                    let inv = f64::from(theta).powf(-((2 * j) as f64) / d as f64);
                    let (s, c) = (f64::from(position) * inv).sin_cos();
                    let (a, bb) = (f64::from(x[base + j]), f64::from(x[base + half + j]));
                    want[base + j] = a * c - bb * s;
                    want[base + half + j] = bb * c + a * s;
                }
            }
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
    }

    /// A sectioned rotation is refused unless its widths cover the rotary pairs exactly, fit
    /// `usize`, and the pairing is half-split.
    #[test]
    fn sectioned_rope_widths_are_checked_against_the_rotary_pairs() {
        let plain = RopeParams::new(8, 10_000.0, &RopeFlavor::Plain).unwrap();
        let field = "rope_scaling.mrope_section";
        assert!(plain.sectioned([1, 2, 1]).is_ok());
        assert_eq!(
            plain.sectioned([1, 1, 1]),
            Err(ParamError::new(
                field,
                ConfigReason::Mismatch {
                    expected: 4,
                    found: 3
                }
            ))
        );
        assert_eq!(
            plain.sectioned([usize::MAX, 1, 1]),
            Err(ParamError::new(field, ConfigReason::Overflow))
        );
        assert_eq!(
            plain.interleaved().sectioned([1, 2, 1]),
            Err(ParamError::new(field, ConfigReason::Unsupported))
        );
    }

    /// SC-001: interleaved RoPE rotates each adjacent (2j, 2j + 1) pair by `theta^(-2j/rotary)`
    /// at the absolute position (f64 reference), full and partial rotary; and it equals the
    /// half-split rotation of the same row with its dims un-permuted (the GGUF Q/K layout).
    #[test]
    fn interleaved_rope_rotates_adjacent_pairs_and_equals_the_unpermuted_half_split() {
        let (heads, t, d, cap) = (2, 3, 8, 16);
        let theta = 10_000.0f32;
        let pos = [5, 6, 11];
        let x = oracle::random(heads * t * d, 22);
        let run = |p: RopeParams, x: &[f32]| {
            let b = Builder::new();
            let ps = b.slot(Slot::Pos, TensorType::new(vec![1, t], DType::I32));
            let xs = b.constant("x", TensorType::f32(vec![1, heads, t, d]));
            let rows = rope_rows(&b, p, ps, cap);
            let y = apply_rope(&b, xs, &rows);
            Fx::new(0)
                .eval(
                    &b.finish(y).unwrap(),
                    &[("x", x)],
                    &[(Slot::Pos, &pos)],
                    &[],
                )
                .0
        };
        for rotary in [d, 4] {
            let plain = RopeParams::new(rotary, theta, &RopeFlavor::Plain).unwrap();
            let got = run(plain.interleaved(), &x);
            let mut want: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
            for h in 0..heads {
                for (i, &p) in pos.iter().enumerate() {
                    let base = (h * t + i) * d;
                    for j in 0..rotary / 2 {
                        let inv = f64::from(theta).powf(-((2 * j) as f64) / rotary as f64);
                        let (s, c) = (f64::from(p) * inv).sin_cos();
                        let (a, bb) = (f64::from(x[base + 2 * j]), f64::from(x[base + 2 * j + 1]));
                        want[base + 2 * j] = a * c - bb * s;
                        want[base + 2 * j + 1] = bb * c + a * s;
                    }
                }
            }
            oracle::assert_matches_f64(&got, &want, 1e-5);

            // Permute each head's rotary dims HF -> GGUF (new 2j = old j, new 2j+1 = old j + half).
            let permute = |v: &[f32]| -> Vec<f32> {
                let mut out = v.to_vec();
                for row in out.chunks_mut(d) {
                    let src = row.to_vec();
                    for j in 0..rotary / 2 {
                        row[2 * j] = src[j];
                        row[2 * j + 1] = src[rotary / 2 + j];
                    }
                }
                out
            };
            assert_eq!(
                run(plain.interleaved(), &permute(&x)),
                permute(&run(plain, &x))
            );
        }
    }

    /// llama.cpp's converter rule for `rope_freqs.weight` (f64): the per-frequency divisor of each
    /// inverse frequency under llama3 scaling.
    fn llama3_factors(rotary: usize, theta: f64, scaling: [f64; 4]) -> Vec<f32> {
        let [factor, low, high, original] = scaling;
        (0..rotary / 2)
            .map(|j| {
                let wavelen =
                    2.0 * std::f64::consts::PI * theta.powf((2 * j) as f64 / rotary as f64);
                let divisor = if wavelen < original / high {
                    1.0
                } else if wavelen > original / low {
                    factor
                } else {
                    let smooth = (original / wavelen - low) / (high - low);
                    1.0 / ((1.0 - smooth) / factor + smooth)
                };
                divisor as f32
            })
            .collect()
    }

    /// llama.cpp's `rope_freqs.weight` of Llama 3.2 (`factor 32`, `low 1`, `high 4`, `8192`) decodes
    /// back to those scaling fields over a 64-wide head at `theta 5e5`, and the decoded RoPE rotates
    /// like the stored factors. Mutation: decode to plain.
    #[test]
    fn llama3_freq_factors_decode_to_the_scaling_that_wrote_them() {
        let (rotary, theta) = (64, 500_000.0f32);
        let factors = llama3_factors(rotary, f64::from(theta), [32.0, 1.0, 4.0, 8192.0]);
        let flavor = llama3_flavor_from_factors(&factors, rotary, theta).unwrap();
        let RopeFlavor::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original,
        } = flavor
        else {
            panic!("{flavor:?}");
        };
        assert_eq!(original, 8192);
        assert!((factor - 32.0).abs() < 1e-4, "{factor}");
        assert!((low_freq_factor - 1.0).abs() < 1e-2, "{low_freq_factor}");
        assert!((high_freq_factor - 4.0).abs() < 1e-2, "{high_freq_factor}");
        let plain = RopeParams::new(rotary, theta, &RopeFlavor::Plain).unwrap();
        assert_ne!(plain.with_freq_factors(&factors).unwrap(), plain);
    }

    /// All-ones factors are plain RoPE; factors that are not a llama3 rescale are refused.
    #[test]
    fn freq_factors_that_are_not_llama3_are_plain_or_refused() {
        assert_eq!(
            llama3_flavor_from_factors(&[1.0; 32], 64, 1e4),
            Ok(RopeFlavor::Plain)
        );
        let refuse = |factors: &[f32]| llama3_flavor_from_factors(factors, 64, 1e4);
        assert_eq!(refuse(&[1.0; 31]), Err(ConfigReason::WrongType));
        assert_eq!(refuse(&[0.5; 32]), Err(ConfigReason::Unsupported));
        let mut stair = vec![1.0f32; 32];
        stair[20..].fill(4.0);
        assert_eq!(refuse(&stair), Err(ConfigReason::Unsupported));
        let wobble: Vec<f32> = (0..32).map(|j| 1.0 + (j % 5) as f32).collect();
        assert_eq!(refuse(&wobble), Err(ConfigReason::Unsupported));
    }

    /// SC-005 (spec 999): each invalid RoPE combination is a typed error naming its field.
    #[test]
    fn rope_params_refuse_each_invalid_combination_by_field() {
        let plain = RopeFlavor::Plain;
        let cases: &[(usize, f32, RopeFlavor<'_>, ParamError)] = &[
            (
                0,
                1e4,
                plain,
                ParamError::new("rotary_dim", ConfigReason::Zero),
            ),
            (
                7,
                1e4,
                plain,
                ParamError::new("rotary_dim", ConfigReason::NotDivisible { by: 2 }),
            ),
            (
                8,
                0.0,
                plain,
                ParamError::new("rope_theta", ConfigReason::NotFinitePositive),
            ),
            (
                8,
                f32::NAN,
                plain,
                ParamError::new("rope_theta", ConfigReason::NotFinitePositive),
            ),
            (
                8,
                1e4,
                RopeFlavor::Linear { factor: -2.0 },
                ParamError::new("rope_scaling.factor", ConfigReason::NotFinitePositive),
            ),
            (
                8,
                1e4,
                RopeFlavor::DynamicNtk {
                    factor: 2.0,
                    original: 0,
                },
                ParamError::new(
                    "rope_scaling.original_max_position_embeddings",
                    ConfigReason::Zero,
                ),
            ),
            (
                8,
                1e4,
                RopeFlavor::LongRope {
                    original: 8,
                    max_positions: 16,
                    short: None,
                    long: None,
                },
                ParamError::new("rope_scaling", ConfigReason::Unsupported),
            ),
        ];
        for (rotary, theta, flavor, want) in cases {
            assert_eq!(
                RopeParams::new(*rotary, *theta, flavor),
                Err(*want),
                "{rotary} {theta} {flavor:?}"
            );
        }
    }

    #[test]
    fn hf_rope_scaling_reads_each_table_flavor_and_refuses_the_rest() {
        let flavor = |v: serde_json::Value| hf_rope_flavor(Some(&v), 4096);
        assert_eq!(hf_rope_flavor(None, 4096), Ok(RopeFlavor::Plain));
        assert_eq!(flavor(json!(null)), Ok(RopeFlavor::Plain));
        assert_eq!(
            flavor(json!({"type": "linear", "factor": 2.0})),
            Ok(RopeFlavor::Linear { factor: 2.0 })
        );
        assert_eq!(
            flavor(json!({"rope_type": "dynamic", "factor": 3.0})),
            Ok(RopeFlavor::DynamicNtk {
                factor: 3.0,
                original: 4096
            })
        );
        assert_eq!(
            flavor(
                json!({"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 1024})
            ),
            Ok(RopeFlavor::Yarn {
                factor: 4.0,
                original: 1024,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: None
            })
        );
        assert_eq!(
            flavor(
                json!({"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 8192})
            ),
            Ok(RopeFlavor::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original: 8192
            })
        );
        let refused = |v, field, reason| {
            assert_eq!(flavor(v), Err(ParamError::new(field, reason)));
        };
        refused(
            json!({"type": "longrope", "short_factor": [1.0]}),
            "rope_scaling.rope_type",
            ConfigReason::Unsupported,
        );
        refused(
            json!({"type": "linear"}),
            "rope_scaling.factor",
            ConfigReason::Missing,
        );
        refused(
            json!({"type": "linear", "factor": "2"}),
            "rope_scaling.factor",
            ConfigReason::WrongType,
        );
        refused(
            json!({"factor": 2.0}),
            "rope_scaling.rope_type",
            ConfigReason::Missing,
        );
        refused(json!(3), "rope_scaling", ConfigReason::WrongType);
    }

    /// `rope_strided_rows` is the rows of `rope_rows` at positions `0, stride, 2 * stride, ..`
    /// (compressed windows' first positions), for both pairings and broadcast over the rows of the
    /// step; `inverse` undoes `apply_rope` at the same positions. Mutation: stride the table by
    /// `stride + 1`; the first window past the first goes red.
    #[test]
    fn strided_rows_are_the_strided_positions_and_inverse_undoes_the_rotation() {
        let (heads, d, cap, stride, count, rows) = (2, 8, 16, 4, 3, 2);
        for p in [
            RopeParams::new(d, 10_000.0, &RopeFlavor::Plain).unwrap(),
            RopeParams::new(d, 10_000.0, &RopeFlavor::Plain)
                .unwrap()
                .interleaved(),
        ] {
            let x = oracle::random(rows * heads * count * d, 41);
            let positions: Vec<i32> = (0..count).map(|w| (w * stride) as i32).collect();
            // `which`: 0 rows at the strided positions, 1 rows gathered at those positions, 2
            // the strided rotation undone by its inverse.
            let run = |which: u8| {
                let b = Builder::new();
                let xs = b.constant("x", TensorType::f32(vec![rows, heads, count, d]));
                let r = if which == 1 {
                    let ps = b.slot(Slot::Pos, TensorType::new(vec![rows, count], DType::I32));
                    rope_rows(&b, p, ps, cap)
                } else {
                    rope_strided_rows(&b, p, cap, stride, count, rows)
                };
                let y = apply_rope(&b, xs, &r);
                let out = if which == 2 {
                    apply_rope(&b, y, &r.inverse(&b))
                } else {
                    y
                };
                let pos: Vec<i32> = positions.iter().chain(&positions).copied().collect();
                Fx::new(0)
                    .eval(
                        &b.finish(out).unwrap(),
                        &[("x", &x)],
                        &[(Slot::Pos, &pos)],
                        &[],
                    )
                    .0
            };
            let (strided, by_pos, back) = (run(0), run(1), run(2));
            oracle::assert_matches_f64(
                &strided,
                &by_pos.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                1e-6,
            );
            assert_ne!(strided, x, "the rotation moves the values");
            oracle::assert_matches_f64(
                &back,
                &x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                1e-5,
            );
        }
    }
}
