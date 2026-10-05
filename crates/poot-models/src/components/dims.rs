//! Reading a decoder's raw config: the field reader every family's `from_raw` shares ([`Fields`])
//! and the dimensions every dense decoder declares ([`DenseDims`]).
//!
//! A checkpoint spells the same fact two ways, an HF `config.json` key and a GGUF metadata key.
//! A [`Field`] names both; [`Fields`] reads whichever the raw config is and reports a malformed
//! value as a typed [`ModelError::Config`] naming the key the checkpoint used. The reader holds no
//! family knowledge: a family passes its own keys, defaults and the family name it reports.

use std::collections::BTreeSet;

use poot_graph_ir::rope_table::RopeFlavor;
use poot_load::gguf::GgufIndex;

use super::attention::AttentionParams;
use super::norm::NormParams;
use super::rope::{RopeParams, hf_rope_flavor};
use crate::model::{ConfigReason, FamilyKey, ModelError};
use crate::registry::RawConfig;

/// One config fact: its HF `config.json` key and its GGUF metadata key (`""` when GGUF has none).
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub hf: &'static str,
    pub gguf: &'static str,
}

impl Field {
    pub const fn new(hf: &'static str, gguf: &'static str) -> Self {
        Self { hf, gguf }
    }

    /// A fact only an HF config carries.
    pub const fn hf(hf: &'static str) -> Self {
        Self { hf, gguf: "" }
    }
}

/// A raw config read for one family.
#[derive(Clone, Copy)]
pub struct Fields<'a> {
    raw: &'a RawConfig<'a>,
    family: FamilyKey,
}

impl<'a> Fields<'a> {
    pub fn new(raw: &'a RawConfig<'a>, family: FamilyKey) -> Self {
        Self { raw, family }
    }

    pub fn family(&self) -> FamilyKey {
        self.family
    }

    pub fn raw(&self) -> &'a RawConfig<'a> {
        self.raw
    }

    pub fn error(&self, field: &'static str, reason: ConfigReason) -> ModelError {
        ModelError::Config {
            family: self.family,
            field,
            reason,
        }
    }

    /// The key `field` has in this config's format.
    pub fn key(&self, field: Field) -> &'static str {
        match self.raw {
            RawConfig::HfJson { .. } => field.hf,
            RawConfig::Gguf(_) => field.gguf,
        }
    }

    fn hf_value(&self, key: &str) -> Option<&'a serde_json::Value> {
        match self.raw {
            RawConfig::HfJson { config, .. } => config.get(key).filter(|v| !v.is_null()),
            RawConfig::Gguf(_) => None,
        }
    }

    fn gguf(&self) -> Option<&'a GgufIndex> {
        match self.raw {
            RawConfig::Gguf(gguf) => Some(gguf),
            RawConfig::HfJson { .. } => None,
        }
    }

    /// An optional unsigned integer: absent is `None`, a non-integer `WrongType`.
    pub fn opt_u64(&self, field: Field) -> Result<Option<u64>, ModelError> {
        let key = self.key(field);
        let value = match self.raw {
            RawConfig::HfJson { .. } => self.hf_value(key).map(|v| v.as_u64()),
            RawConfig::Gguf(_) => self
                .gguf()
                .filter(|_| !key.is_empty())
                .and_then(|g| g.get(key))
                .map(|v| v.as_u64()),
        };
        value
            .map(|v| v.ok_or(self.error(key, ConfigReason::WrongType)))
            .transpose()
    }

    /// An optional float.
    pub fn opt_f32(&self, field: Field) -> Result<Option<f32>, ModelError> {
        let key = self.key(field);
        let value = match self.raw {
            RawConfig::HfJson { .. } => self.hf_value(key).map(|v| v.as_f64().map(|v| v as f32)),
            RawConfig::Gguf(_) => self
                .gguf()
                .filter(|_| !key.is_empty())
                .and_then(|g| g.get(key))
                .map(|v| v.as_f32()),
        };
        value
            .map(|v| v.ok_or(self.error(key, ConfigReason::WrongType)))
            .transpose()
    }

    /// An optional boolean (HF only: a GGUF has none of these).
    pub fn opt_bool(&self, field: Field) -> Result<Option<bool>, ModelError> {
        let key = self.key(field);
        let value = match self.raw {
            RawConfig::HfJson { .. } => self.hf_value(key).map(|v| v.as_bool()),
            RawConfig::Gguf(_) => self
                .gguf()
                .filter(|_| !key.is_empty())
                .and_then(|g| g.get(key))
                .map(|v| v.as_bool()),
        };
        value
            .map(|v| v.ok_or(self.error(key, ConfigReason::WrongType)))
            .transpose()
    }

    /// A non-zero count; absent is `default`, or `Missing` without one.
    pub fn count(&self, field: Field, default: Option<usize>) -> Result<usize, ModelError> {
        let key = self.key(field);
        let n = match self.opt_u64(field)? {
            None => default.ok_or(self.error(key, ConfigReason::Missing))?,
            Some(n) => usize::try_from(n).map_err(|_| self.error(key, ConfigReason::WrongType))?,
        };
        if n == 0 {
            return Err(self.error(key, ConfigReason::Zero));
        }
        Ok(n)
    }

    pub fn float(&self, field: Field, default: f32) -> Result<f32, ModelError> {
        Ok(self.opt_f32(field)?.unwrap_or(default))
    }

    pub fn flag(&self, field: Field, default: bool) -> Result<bool, ModelError> {
        Ok(self.opt_bool(field)?.unwrap_or(default))
    }

    /// A token id (`bos_token_id`), when the config names one.
    pub fn token(&self, field: Field) -> Result<Option<u32>, ModelError> {
        let key = self.key(field);
        self.opt_u64(field)?
            .map(|id| u32::try_from(id).map_err(|_| self.error(key, ConfigReason::WrongType)))
            .transpose()
    }

    /// The vocabulary size: `vocab_size`, or the length of a GGUF's token list.
    pub fn vocab(&self) -> Result<usize, ModelError> {
        match self.raw {
            RawConfig::HfJson { .. } => self.count(Field::hf("vocab_size"), None),
            RawConfig::Gguf(gguf) => {
                let key = "tokenizer.ggml.tokens";
                let tokens = gguf
                    .get(key)
                    .ok_or(self.error(key, ConfigReason::Missing))?
                    .as_array()
                    .ok_or(self.error(key, ConfigReason::WrongType))?;
                if tokens.is_empty() {
                    return Err(self.error(key, ConfigReason::Zero));
                }
                Ok(tokens.len())
            }
        }
    }

    /// The end-of-sequence set the checkpoint declares, or `default` when it declares none.
    pub fn eos(&self, default: Option<u32>) -> Result<BTreeSet<u32>, ModelError> {
        let mut eos = self.raw.eos_token_ids(self.family)?;
        if eos.is_empty() {
            eos.extend(default);
        }
        Ok(eos)
    }

    pub fn bos(&self) -> Result<Option<u32>, ModelError> {
        self.token(Field::new("bos_token_id", "tokenizer.ggml.bos_token_id"))
    }

    /// The base of an HF config that nests its RoPE block (`rope_parameters.rope_theta`).
    pub fn hf_rope_theta(&self) -> Result<Option<f32>, ModelError> {
        const KEY: &str = "rope_parameters.rope_theta";
        self.hf_value("rope_parameters")
            .and_then(|p| p.get("rope_theta"))
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_f64()
                    .map(|v| v as f32)
                    .ok_or(self.error(KEY, ConfigReason::WrongType))
            })
            .transpose()
    }

    /// The HF `rope_scaling` flavor (plain when absent); `RopeFlavor::Plain` for a GGUF, whose
    /// scaling the family reads from its own keys.
    pub fn hf_rope_flavor(&self, max_positions: usize) -> Result<RopeFlavor<'static>, ModelError> {
        let scaling = self
            .hf_value("rope_scaling")
            .or_else(|| self.hf_value("rope_parameters"));
        hf_rope_flavor(scaling, max_positions).map_err(|e| e.for_family(self.family))
    }
}

/// The GGUF keys of a decoder family under its architecture prefix (`"qwen2"`, `"llama"`).
#[derive(Clone, Copy, Debug)]
pub struct DenseKeys {
    pub width: &'static str,
    pub inter: &'static str,
    pub layers: &'static str,
    pub heads: &'static str,
    pub kv_heads: &'static str,
    pub head_dim: &'static str,
    pub eps: &'static str,
    pub max_positions: &'static str,
    pub theta: &'static str,
    pub rope_scaling_type: &'static str,
    pub rope_scaling_factor: &'static str,
}

/// [`DenseKeys`] for a literal architecture prefix: `dense_keys!("llama")`.
macro_rules! dense_keys {
    ($arch:literal) => {
        $crate::components::dims::DenseKeys {
            width: concat!($arch, ".embedding_length"),
            inter: concat!($arch, ".feed_forward_length"),
            layers: concat!($arch, ".block_count"),
            heads: concat!($arch, ".attention.head_count"),
            kv_heads: concat!($arch, ".attention.head_count_kv"),
            head_dim: concat!($arch, ".attention.key_length"),
            eps: concat!($arch, ".attention.layer_norm_rms_epsilon"),
            max_positions: concat!($arch, ".context_length"),
            theta: concat!($arch, ".rope.freq_base"),
            rope_scaling_type: concat!($arch, ".rope.scaling.type"),
            rope_scaling_factor: concat!($arch, ".rope.scaling.factor"),
        }
    };
}
pub(crate) use dense_keys;

/// What a family fixes about its dense decoder's config: its name, its GGUF keys and the defaults a
/// checkpoint may omit (HF's config-class defaults, llama.cpp's GGUF defaults).
#[derive(Clone, Copy, Debug)]
pub struct DenseSpec {
    pub family: FamilyKey,
    pub gguf: DenseKeys,
    pub hf_theta: f32,
    pub gguf_theta: f32,
    pub eps: f32,
    pub eos: Option<u32>,
}

/// A dense decoder's dimensions, every one non-zero and consistent (`width` divisible by `heads`
/// unless the config states `head_dim`, `kv_heads` dividing `heads`).
#[derive(Clone, Debug, PartialEq)]
pub struct DenseDims {
    pub vocab: usize,
    pub width: usize,
    pub inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub max_positions: usize,
    pub theta: f32,
    pub flavor: RopeFlavor<'static>,
    /// The checkpoint must carry its own head: HF `tie_word_embeddings: false`.
    pub head_required: bool,
    pub eos: BTreeSet<u32>,
    pub bos: Option<u32>,
}

impl DenseDims {
    pub fn read(fields: &Fields<'_>, spec: &DenseSpec) -> Result<Self, ModelError> {
        let k = &spec.gguf;
        let field = |hf, gguf| Field::new(hf, gguf);
        let width_f = field("hidden_size", k.width);
        let heads_f = field("num_attention_heads", k.heads);
        let kv_f = field("num_key_value_heads", k.kv_heads);
        let head_dim_f = field("head_dim", k.head_dim);
        let max_f = field("max_position_embeddings", k.max_positions);
        let theta_f = field("rope_theta", k.theta);
        let eps_f = field("rms_norm_eps", k.eps);

        let vocab = fields.vocab()?;
        let width = fields.count(width_f, None)?;
        let heads = fields.count(heads_f, None)?;
        let kv_heads = fields.count(kv_f, Some(heads))?;
        let max_positions = fields.count(max_f, None)?;
        let head_dim = match fields.opt_u64(head_dim_f)? {
            Some(0) => return Err(fields.error(fields.key(head_dim_f), ConfigReason::Zero)),
            Some(d) => usize::try_from(d)
                .map_err(|_| fields.error(fields.key(head_dim_f), ConfigReason::WrongType))?,
            None => {
                if !width.is_multiple_of(heads) {
                    return Err(fields.error(
                        fields.key(width_f),
                        ConfigReason::NotDivisible { by: heads },
                    ));
                }
                width / heads
            }
        };
        if !heads.is_multiple_of(kv_heads) {
            return Err(fields.error(
                fields.key(heads_f),
                ConfigReason::NotDivisible { by: kv_heads },
            ));
        }
        if u32::try_from(max_positions).is_err() {
            return Err(fields.error(
                fields.key(max_f),
                ConfigReason::Exceeds {
                    max: u32::MAX as usize,
                },
            ));
        }
        let (flavor, head_required, theta_default) = match fields.raw() {
            RawConfig::HfJson { config, .. } => {
                let tied = match config.get("tie_word_embeddings").filter(|v| !v.is_null()) {
                    None => false,
                    Some(v) => v
                        .as_bool()
                        .ok_or(fields.error("tie_word_embeddings", ConfigReason::WrongType))?,
                };
                (fields.hf_rope_flavor(max_positions)?, !tied, spec.hf_theta)
            }
            RawConfig::Gguf(gguf) => (
                gguf_rope_flavor(fields, gguf, spec)?,
                false,
                spec.gguf_theta,
            ),
        };
        Ok(Self {
            vocab,
            width,
            inter: fields.count(field("intermediate_size", k.inter), None)?,
            layers: fields.count(field("num_hidden_layers", k.layers), None)?,
            heads,
            kv_heads,
            head_dim,
            eps: fields.float(eps_f, spec.eps)?,
            max_positions,
            theta: match fields.opt_f32(theta_f)? {
                Some(theta) => theta,
                None => fields.hf_rope_theta()?.unwrap_or(theta_default),
            },
            flavor,
            head_required,
            eos: fields.eos(spec.eos)?,
            bos: fields.bos()?,
        })
    }
}

impl DenseDims {
    /// The RMSNorm epsilon, checked.
    pub fn norm(&self, fields: &Fields<'_>, spec: &DenseSpec) -> Result<NormParams, ModelError> {
        let key = fields.key(Field::new("rms_norm_eps", spec.gguf.eps));
        NormParams::new(self.eps).map_err(|_| fields.error(key, ConfigReason::NotFinitePositive))
    }

    /// RoPE over the leading `rotary` dims of a head, at this config's base and scaling.
    pub fn rope(
        &self,
        fields: &Fields<'_>,
        spec: &DenseSpec,
        rotary: usize,
    ) -> Result<RopeParams, ModelError> {
        RopeParams::new(rotary, self.theta, &self.flavor).map_err(|e| {
            if e.field == "rope_theta" {
                fields.error(
                    fields.key(Field::new("rope_theta", spec.gguf.theta)),
                    e.reason,
                )
            } else {
                e.for_family(fields.family())
            }
        })
    }

    /// The base attention parameters over a full-width rotation: `heads` over `kv_heads`
    /// `head_dim`-wide heads, with a QKV bias when `qkv_bias`.
    pub fn attention(
        &self,
        fields: &Fields<'_>,
        spec: &DenseSpec,
        qkv_bias: bool,
    ) -> Result<AttentionParams, ModelError> {
        let rope = self.rope(fields, spec, self.head_dim)?;
        AttentionParams::new(self.heads, self.kv_heads, self.head_dim, qkv_bias, rope)
            .map_err(|e| e.for_family(fields.family()))
    }
}

/// A GGUF's RoPE scaling: absent or `none` is plain, `linear` reads `rope.scaling.factor`; any
/// other type is refused (llama.cpp carries those as tensors a table cannot hold).
fn gguf_rope_flavor(
    fields: &Fields<'_>,
    gguf: &GgufIndex,
    spec: &DenseSpec,
) -> Result<RopeFlavor<'static>, ModelError> {
    let key = spec.gguf.rope_scaling_type;
    let Some(kind) = gguf.get(key) else {
        return Ok(RopeFlavor::Plain);
    };
    match kind.as_str() {
        Some("none") => Ok(RopeFlavor::Plain),
        Some("linear") => {
            let key = spec.gguf.rope_scaling_factor;
            let factor = gguf
                .get(key)
                .ok_or(fields.error(key, ConfigReason::Missing))?
                .as_f32()
                .ok_or(fields.error(key, ConfigReason::WrongType))?;
            Ok(RopeFlavor::Linear { factor })
        }
        Some(_) => Err(fields.error(key, ConfigReason::Unsupported)),
        None => Err(fields.error(key, ConfigReason::WrongType)),
    }
}
