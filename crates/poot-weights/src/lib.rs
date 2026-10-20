//! The weight vocabulary and the weight map (card 562a; POOT-1258).
//!
//! [`WeightRole`] says what a weight is for, independent of any checkpoint's tensor names, and
//! [`WeightId`] (layer and role) is the one identity of a mapped weight: the graph const name
//! ([`WeightId::const_name`]), the key of a model's [`WeightMap`], and what a binder looks a weight up by.
//! A [`WeightMap`] is a model's table from a [`WeightId`] to a [`WeightView`] over `poot_quant`'s
//! `WeightStore` entries and the [`WeightHandle`] a tracer sees (format and logical shape, no bytes).
//!
//! It is its own crate, not part of `poot-quant`, because a role names what a model family has (Mamba2,
//! GDN, KDA, DSA, MHC, ... roles), while `poot-quant` is storage formats, `StoredBytes`, `DenseWeight`,
//! `PackedPayload` and the store: a new architecture's roles edit this crate, not the quantization
//! foundation the kernel generator also builds on.

mod map;
mod role;

pub use map::{
    HandleFormat, LoadReport, WeightHandle, WeightMap, WeightMapBuilder, WeightMapError, WeightView,
};
pub use role::{
    AttnRole, CompressRole, DsaRole, ExpertProj, FfnRole, GdnRole, HyperRole, HyperSite, KdaRole,
    LoraRole, LoraTarget, Mamba2Role, MhcRole, MlaRole, MoeRole, MtpRole, NormRole, PleRole,
    QsaRole, WeightId, WeightRole,
};
