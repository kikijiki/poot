//! Input modalities and which of them a loaded model can take.

use crate::core::runner::Runner;

/// A kind of request input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
    Video,
    Audio,
}

impl Modality {
    /// The lowercase name used in messages.
    pub const fn name(self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Video => "video",
            Modality::Audio => "audio",
        }
    }
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl Runner {
    /// Whether this model has a front end for `modality`, that is, a way to turn that input into
    /// embeddings. A `Runner` is a text decoder: its only front end is the token embedding, so it takes
    /// [`Modality::Text`] and nothing else. The vision-language model (`vlm::VlmRunner`) is a separate
    /// type with its own image encoder and is not a `Runner`. A model that gains a front end reports it
    /// here; a caller decides on this answer, never on the model family.
    pub fn accepts_modality(&self, modality: Modality) -> bool {
        modality == Modality::Text
    }
}
