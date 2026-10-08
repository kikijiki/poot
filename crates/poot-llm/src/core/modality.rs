//! Input modalities and which of them a loaded model can take.

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
