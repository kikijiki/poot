//! The chat template of the Qwen2.5-VL fixtures: the shape of the released template, reduced to the role
//! markers and the media parts.

/// A chat template that prints each message as its role, then its content parts: the three vision
/// markers around a pad for an image or a video part, the text of a text part. The conversation is
/// passed as parts (the released template's list branch), so the pads are printed by the template and
/// not by the message text.
pub const CHAT_TEMPLATE: &str = "{% for message in messages %}{{ message.role }} {% for part in message.content %}{% if part.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif part.type == 'video' %}<|vision_start|><|video_pad|><|vision_end|>{% else %}{{ part.text }}{% endif %}{% endfor %}\n{% endfor %}";
