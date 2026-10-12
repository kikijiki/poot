//! Integrity pins for the files a checkpoint is loaded from: a length and SHA-256 digest a caller
//! checks a file against ([`FilePin`], [`check_pin_reader`]), and a JSON reader that refuses a
//! repeated object key ([`parse_json_unique_keys`]).

use std::fmt;

use ring::digest::{Context, SHA256};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

/// The read size `check_pin_reader` hashes in.
const HASH_BUFFER_BYTES: usize = 4096;

/// A failed integrity check.
#[derive(Debug, thiserror::Error)]
pub enum PackedSafetensorsError {
    #[error("{file}: I/O failed: {error}")]
    Io {
        file: String,
        #[source]
        error: std::io::Error,
    },
    #[error("{file}: digest mismatch")]
    DigestMismatch { file: String },
    #[error("authenticated artifact changed before publication: {file}")]
    ArtifactChanged { file: String },
}

/// An exact SHA-256 digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

impl From<[u8; 32]> for Sha256Digest {
    fn from(value: [u8; 32]) -> Self {
        Self::new(value)
    }
}

/// Hash a byte slice using the artifact identity algorithm.
#[cfg(any(test, feature = "test-support"))]
pub fn sha256_digest(bytes: &[u8]) -> Sha256Digest {
    let digest = ring::digest::digest(&SHA256, bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_ref());
    Sha256Digest(out)
}

/// One resident file's optional integrity pin (card 544, ADR-0103 decision 4): the exact length
/// and digest a manifest names for it. A manifest is optional integrity data, never the admission
/// gate - a file with no pin is still admitted, by whatever structural (capability) check the
/// caller applies to its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilePin {
    pub length: usize,
    pub sha256: Sha256Digest,
}

/// Check a file too large to hold against its [`FilePin`]: the reader is hashed in fixed chunks, once,
/// and its total length and digest compared with `pin`. A short or long file is `ArtifactChanged`, a
/// different digest `DigestMismatch`, a failed read `Io`.
pub fn check_pin_reader(
    mut reader: impl std::io::Read,
    pin: FilePin,
    name: &str,
) -> Result<(), PackedSafetensorsError> {
    let mut context = Context::new(&SHA256);
    let mut buffer = [0u8; HASH_BUFFER_BYTES];
    let mut total = 0usize;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| PackedSafetensorsError::Io {
                file: name.to_string(),
                error,
            })?;
        if read == 0 {
            break;
        }
        context.update(&buffer[..read]);
        total += read;
    }
    if total != pin.length {
        return Err(PackedSafetensorsError::ArtifactChanged {
            file: name.to_string(),
        });
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(context.finish().as_ref());
    if Sha256Digest(digest) != pin.sha256 {
        return Err(PackedSafetensorsError::DigestMismatch {
            file: name.to_string(),
        });
    }
    Ok(())
}

struct UniqueJson(serde_json::Value);

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueJsonVisitor;

        impl<'de> Visitor<'de> for UniqueJsonVisitor {
            type Value = UniqueJson;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value with unique object keys")
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Bool(value)))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Number(value.into())))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Number(value.into())))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let number = serde_json::Number::from_f64(value)
                    .ok_or_else(|| E::custom("non-finite JSON number"))?;
                Ok(UniqueJson(serde_json::Value::Number(number)))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.visit_string(value.to_string())
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::String(value)))
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Null))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Null))
            }

            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: Deserializer<'de>,
            {
                UniqueJson::deserialize(deserializer)
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(UniqueJson(value)) = sequence.next_element()? {
                    values.push(value);
                }
                Ok(UniqueJson(serde_json::Value::Array(values)))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(<A::Error as de::Error>::custom(format!(
                            "duplicate JSON key {key:?}"
                        )));
                    }
                    let UniqueJson(value) = map.next_value()?;
                    values.insert(key, value);
                }
                Ok(UniqueJson(serde_json::Value::Object(values)))
            }
        }

        deserializer.deserialize_any(UniqueJsonVisitor)
    }
}

/// `bytes` as a JSON value, refusing an object that repeats a key at any depth: `serde_json::Value`
/// alone keeps the last of a repeated key, so a config's earlier setting would be dropped unseen.
pub fn parse_json_unique_keys(bytes: &[u8]) -> Result<serde_json::Value, serde_json::Error> {
    serde_json::from_slice::<UniqueJson>(bytes).map(|value| value.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The streamed pin check: an input several chunks long with a partial last chunk, one exactly a
    /// chunk long, and an empty file are admitted under their own pin, refused as `DigestMismatch`
    /// when one byte differs and as `ArtifactChanged` under a wrong length. Mutation: drop the last
    /// byte of every read from the hash in `check_pin_reader` (`context.update(&buffer[..read - 1])`);
    /// the multi-chunk row fails with `DigestMismatch` for the file it should admit.
    #[test]
    fn the_streamed_pin_check_admits_its_own_pin_and_refuses_a_changed_file() {
        for len in [0, HASH_BUFFER_BYTES, 3 * HASH_BUFFER_BYTES + 17] {
            let bytes: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
            let pin = FilePin {
                length: len,
                sha256: sha256_digest(&bytes),
            };
            check_pin_reader(bytes.as_slice(), pin, "f").unwrap_or_else(|e| panic!("{len}: {e:?}"));
            let mut other = bytes.clone();
            if let Some(last) = other.last_mut() {
                *last ^= 1;
                let error = check_pin_reader(other.as_slice(), pin, "f").unwrap_err();
                assert!(matches!(
                    error,
                    PackedSafetensorsError::DigestMismatch { .. }
                ));
            }
            let short = FilePin {
                length: len + 1,
                ..pin
            };
            let error = check_pin_reader(bytes.as_slice(), short, "f").unwrap_err();
            assert!(matches!(
                error,
                PackedSafetensorsError::ArtifactChanged { .. }
            ));
        }
    }

    /// A repeated object key is refused at any depth, where `serde_json::Value` alone would keep the
    /// last one. Mutation: replace the `values.contains_key(&key)` check in `UniqueJson`'s `visit_map`
    /// with `false`; the nested row parses and the assertion goes red.
    #[test]
    fn a_repeated_json_key_is_refused_at_any_depth() {
        parse_json_unique_keys(br#"{"a":1,"b":{"c":[1,2]}}"#).unwrap();
        for text in [
            r#"{"a":1,"a":2}"#,
            r#"{"a":{"b":1,"b":2}}"#,
            r#"[{"k":0,"k":0}]"#,
        ] {
            let error = parse_json_unique_keys(text.as_bytes()).unwrap_err();
            assert!(
                error.to_string().contains("duplicate JSON key"),
                "{text}: {error}"
            );
        }
    }
}
