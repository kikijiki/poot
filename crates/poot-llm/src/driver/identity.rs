//! What a run says about what it ran on (POOT-1334): the build of the binary, the adapter and driver of the
//! device, the digest of the weights it loaded and the dtype it computes in. The benchmark runner states these
//! on every result line and the server states them on its identity route, both through [`RunIdentity`], so
//! the two cannot name different fields.

use std::collections::BTreeMap;

use poot_build_identity::BuildIdentity;
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_quant::{PayloadReadError, ReadError};
use poot_runtime_common::DeviceIdentity;

/// The dtype the engine computes in whatever the weights are stored in: matmuls accumulate in f32 and the
/// activations are f32. Reported beside the resident weight dtype so a row never reads a stored dtype as a
/// compute one.
pub const COMPUTE_DTYPE: &str = "f32";

/// Why a checkpoint has no identity to state.
#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    /// A dense tensor's bytes could not be read for the digest.
    #[error("read a weight to digest the checkpoint: {0}")]
    Read(#[from] ReadError),
    /// A packed tensor's bytes could not be read, or failed their check, for the digest.
    #[error("read a packed weight to digest the checkpoint: {0}")]
    Packed(#[from] PayloadReadError),
    /// The store holds no weight, so it holds no dtype.
    #[error("the store holds no weight")]
    EmptyStore,
}

/// What a run ran on. [`RunIdentity::fields`] is the one list of field names and values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunIdentity {
    /// The binary's build.
    pub build: BuildIdentity,
    /// The registry family the checkpoint resolved to, as the registry reports it.
    pub family: String,
    /// The dtype most of the stored weight bytes are in ([`resident_precision`]): what the checkpoint holds,
    /// never what a request named.
    pub precision: String,
    /// The adapter the executor reported for the device it opened.
    pub adapter: String,
    /// The driver stack the executor's runtime reported.
    pub driver_version: String,
    /// `blake3:` and the digest of the weights as the driver holds them ([`checkpoint_digest`]).
    pub checkpoint_digest: String,
    /// The dtype the engine computes in ([`COMPUTE_DTYPE`]).
    pub compute_dtype: &'static str,
}

impl RunIdentity {
    /// The identity of a run of `build` on `device` over `store`: reads every stored weight once.
    pub fn of(
        build: BuildIdentity,
        family: String,
        device: DeviceIdentity,
        store: &WeightStore,
    ) -> Result<Self, IdentityError> {
        Ok(Self {
            build,
            family,
            precision: resident_precision(store)?,
            adapter: device.adapter,
            driver_version: device.driver,
            checkpoint_digest: checkpoint_digest(store)?,
            compute_dtype: COMPUTE_DTYPE,
        })
    }

    /// The field names and their values, build first, then the family, the stored precision, the device, the checkpoint
    /// digest and the compute dtype: the one list of what a run states about itself.
    pub fn fields(&self) -> [(&'static str, &str); 10] {
        let [sha, profile, features, rustc] = self.build.fields();
        [
            sha,
            profile,
            features,
            rustc,
            ("family", &self.family),
            ("precision", &self.precision),
            ("adapter", &self.adapter),
            ("driver_version", &self.driver_version),
            ("checkpoint_digest", &self.checkpoint_digest),
            ("compute_dtype", self.compute_dtype),
        ]
    }

    /// The identity as one JSON object of [`Self::fields`].
    pub fn to_json(&self) -> serde_json::Value {
        self.fields()
            .into_iter()
            .map(|(name, value)| (name.to_string(), serde_json::Value::from(value)))
            .collect::<serde_json::Map<_, _>>()
            .into()
    }
}

/// A stored weight's dtype as a precision name: the checkpoint dtype of a dense tensor (`bf16`), the format of
/// a packed one without its parameters (`q8_0`, `e4m3block128`). The digest names each weight by it too.
pub fn stored_name(entry: &WeightEntry) -> String {
    match entry {
        WeightEntry::Dense(dense) => dense.dtype().to_string(),
        WeightEntry::Packed(payload) => {
            let format = format!("{:?}", payload.weight().format());
            format
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or_default()
                .to_lowercase()
        }
    }
}

/// The precision the store holds: the dtype most of its stored bytes are in.
pub fn resident_precision(store: &WeightStore) -> Result<String, IdentityError> {
    let mut bytes = BTreeMap::<String, usize>::new();
    for (_, entry) in store.iter() {
        *bytes.entry(stored_name(entry)).or_default() += entry.stored_bytes();
    }
    bytes
        .into_iter()
        .max_by_key(|&(_, stored)| stored)
        .map(|(name, _)| name)
        .ok_or(IdentityError::EmptyStore)
}

/// A digest of the checkpoint's content as the driver holds it: every weight's name, shape, stored dtype and
/// stored bytes in key order. Two checkpoints with one name but different weights digest differently. The
/// bytes are read by positional reads into one bounded buffer, several threads at a time, and hashed across
/// threads, so the digest maps none of the checkpoint in and a cold file is read at the disk's queue depth.
pub fn checkpoint_digest(store: &WeightStore) -> Result<String, IdentityError> {
    /// Bytes hashed per round.
    const CHUNK: usize = 64 << 20;
    /// Reader threads per round.
    const READERS: usize = 8;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; CHUNK];
    for (key, entry) in store.iter() {
        hasher.update(key.to_string().as_bytes());
        hasher.update(format!("{:?}", entry.shape()).as_bytes());
        hasher.update(stored_name(entry).as_bytes());
        match entry {
            WeightEntry::Dense(dense) => {
                let bytes = dense.bytes();
                for start in (0..bytes.len()).step_by(CHUNK) {
                    let end = bytes.len().min(start + CHUNK);
                    let chunk = &mut buffer[..end - start];
                    poot_quant::read_split(chunk, start, READERS, |range, part| {
                        bytes.read_at(range, part)
                    })?;
                    hasher.update_rayon(chunk);
                }
            }
            WeightEntry::Packed(payload) => {
                for role in payload.weight().sources() {
                    let len = payload.weight().source_bytes(role);
                    for start in (0..len).step_by(CHUNK) {
                        let end = len.min(start + CHUNK);
                        let chunk = &mut buffer[..end - start];
                        poot_quant::read_split(chunk, start, READERS, |range, part| {
                            payload.read_verified(role, range, part)
                        })?;
                        hasher.update_rayon(chunk);
                    }
                }
            }
        }
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}
