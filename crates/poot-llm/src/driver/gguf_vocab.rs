//! A GGUF header's tokenizer metadata as [`GgufVocab`], the plain data `poot-text` builds a codec from.
//! `poot-text` reads no GGUF: this is the one place the `tokenizer.ggml.*` keys leave the header index.

use poot_load::gguf::{GgufIndex, GgufValue};
use poot_text::{GgufVocab, TextError};

/// The vocabulary `index` carries. A key the file lacks is `None`; a key that is present but of the wrong
/// type (a non-string token, a score that is not a float, an id beyond `u32`) is a [`TextError`] naming the
/// key, never a default or a dropped array: it would tokenize differently from the file's intent.
pub(crate) fn gguf_vocab(index: &GgufIndex) -> Result<GgufVocab, TextError> {
    fn ill_typed(key: &str, want: &str) -> TextError {
        TextError::Msg(format!("GGUF `{key}` is not {want}"))
    }
    fn scalar<T>(
        index: &GgufIndex,
        key: &str,
        want: &str,
        read: fn(&GgufValue) -> Option<T>,
    ) -> Result<Option<T>, TextError> {
        index
            .get(key)
            .map(|v| read(v).ok_or_else(|| ill_typed(key, want)))
            .transpose()
    }
    fn array<T>(
        index: &GgufIndex,
        key: &str,
        want: &str,
        read: fn(&GgufValue) -> Option<T>,
    ) -> Result<Option<Vec<T>>, TextError> {
        index
            .get(key)
            .map(|v| {
                v.as_array()
                    .ok_or_else(|| ill_typed(key, "an array"))?
                    .iter()
                    .map(|e| read(e).ok_or_else(|| ill_typed(key, want)))
                    .collect()
            })
            .transpose()
    }
    let string = |key: &str| scalar(index, key, "a string", |v| v.as_str().map(str::to_string));
    let id = |key: &str| {
        scalar(index, key, "a u32", |v| {
            v.as_u64().and_then(|v| u32::try_from(v).ok())
        })
    };
    let flag = |key: &str| scalar(index, key, "a bool", GgufValue::as_bool);
    let strings = |key: &str| {
        array(index, key, "an array of strings", |v| {
            v.as_str().map(str::to_string)
        })
    };
    Ok(GgufVocab {
        model: string("tokenizer.ggml.model")?,
        pre: string("tokenizer.ggml.pre")?,
        tokens: strings("tokenizer.ggml.tokens")?,
        scores: array(
            index,
            "tokenizer.ggml.scores",
            "an array of floats",
            GgufValue::as_f32,
        )?,
        token_types: array(index, "tokenizer.ggml.token_type", "an array of u32", |v| {
            v.as_u64().and_then(|t| u32::try_from(t).ok())
        })?,
        merges: strings("tokenizer.ggml.merges")?,
        bos_token_id: id("tokenizer.ggml.bos_token_id")?,
        eos_token_id: id("tokenizer.ggml.eos_token_id")?,
        unknown_token_id: id("tokenizer.ggml.unknown_token_id")?,
        add_bos_token: flag("tokenizer.ggml.add_bos_token")?,
        add_eos_token: flag("tokenizer.ggml.add_eos_token")?,
        add_space_prefix: flag("tokenizer.ggml.add_space_prefix")?,
        chat_template: string("tokenizer.chat_template")?,
    })
}

#[cfg(test)]
mod tests {
    use poot_load::gguf::{GgufIndex, GgufValue, write_gguf};
    use poot_quant::StoredBytes;

    use super::gguf_vocab;

    fn strings(items: &[&str]) -> GgufValue {
        GgufValue::Array(
            items
                .iter()
                .map(|s| GgufValue::Str(s.to_string()))
                .collect(),
        )
    }

    fn index(kvs: &[(&str, GgufValue)]) -> GgufIndex {
        GgufIndex::parse(&StoredBytes::from(write_gguf(kvs, &[]))).expect("parse the fixture")
    }

    /// Every key the vocabulary names lands in its field with the file's value (the wrong-versus-right is
    /// a swapped or dropped key: each field here has a distinct value).
    #[test]
    fn every_tokenizer_key_lands_in_its_field() {
        let vocab = gguf_vocab(&index(&[
            ("general.architecture", GgufValue::Str("llama".into())),
            ("tokenizer.ggml.model", GgufValue::Str("llama".into())),
            ("tokenizer.ggml.pre", GgufValue::Str("llama-bpe".into())),
            ("tokenizer.ggml.tokens", strings(&["<unk>", "<s>", "a"])),
            ("tokenizer.ggml.merges", strings(&["a b"])),
            (
                "tokenizer.ggml.scores",
                GgufValue::Array(vec![GgufValue::F32(0.5), GgufValue::F32(-1.0)]),
            ),
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array(vec![GgufValue::U32(2), GgufValue::U32(3)]),
            ),
            ("tokenizer.ggml.bos_token_id", GgufValue::U32(1)),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(2)),
            ("tokenizer.ggml.unknown_token_id", GgufValue::U32(0)),
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(true)),
            ("tokenizer.ggml.add_eos_token", GgufValue::Bool(false)),
            ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
            ("tokenizer.chat_template", GgufValue::Str("{{ x }}".into())),
        ]))
        .unwrap();
        assert_eq!(vocab.model.as_deref(), Some("llama"));
        assert_eq!(vocab.pre.as_deref(), Some("llama-bpe"));
        assert_eq!(
            vocab.tokens,
            Some(vec!["<unk>".into(), "<s>".into(), "a".into()])
        );
        assert_eq!(vocab.merges, Some(vec!["a b".to_string()]));
        assert_eq!(vocab.scores, Some(vec![0.5, -1.0]));
        assert_eq!(vocab.token_types, Some(vec![2, 3]));
        assert_eq!(vocab.bos_token_id, Some(1));
        assert_eq!(vocab.eos_token_id, Some(2));
        assert_eq!(vocab.unknown_token_id, Some(0));
        assert_eq!(vocab.add_bos_token, Some(true));
        assert_eq!(vocab.add_eos_token, Some(false));
        assert_eq!(vocab.add_space_prefix, Some(false));
        assert_eq!(vocab.chat_template.as_deref(), Some("{{ x }}"));
    }

    /// A header with no tokenizer keys is all `None`: the text services then refuse it by name, never
    /// default a key the file did not carry.
    #[test]
    fn a_header_without_tokenizer_keys_is_all_none() {
        let vocab = gguf_vocab(&index(&[(
            "general.architecture",
            GgufValue::Str("llama".into()),
        )]))
        .unwrap();
        assert!(vocab.model.is_none() && vocab.tokens.is_none() && vocab.merges.is_none());
        assert!(vocab.bos_token_id.is_none() && vocab.add_bos_token.is_none());
        assert!(vocab.chat_template.is_none());
    }

    /// A key that is present with the wrong type is a refusal naming the key; before, a negative token type
    /// dropped the whole vector, silently leaving control tokens unregistered. (The GGUF reader itself
    /// refuses a tokens, merges or scores array of the wrong element type.)
    #[test]
    fn a_present_but_ill_typed_key_is_refused_naming_the_key() {
        for (key, value) in [
            (
                "tokenizer.ggml.token_type",
                GgufValue::Array(vec![GgufValue::I32(-1)]),
            ),
            ("tokenizer.ggml.merges", GgufValue::Str("a b".into())),
            ("tokenizer.ggml.bos_token_id", GgufValue::Str("1".into())),
            ("tokenizer.ggml.add_bos_token", GgufValue::U32(1)),
            ("tokenizer.ggml.model", GgufValue::U32(1)),
        ] {
            let err = gguf_vocab(&index(&[(key, value)])).unwrap_err();
            assert!(err.to_string().contains(key), "{key}: {err}");
        }
    }
}
