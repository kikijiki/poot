//! The kinds of non-text content a chat request part can carry and which of them the server can lower.

/// A kind of non-text request input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Modality {
    /// A still image (PNG or JPEG bytes), lowered through an image front end.
    Image,
    /// A video clip, sampled to frames and lowered through a video-capable front end.
    Video,
    /// Audio samples, for a model with an audio tower.
    Audio,
}

impl Modality {
    /// The lowercase name used in messages.
    pub(crate) const fn name(self) -> &'static str {
        match self {
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
