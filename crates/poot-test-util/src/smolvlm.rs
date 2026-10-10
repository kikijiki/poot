//! The chat template of the SmolVLM checkpoint, for the fixtures that stand in for it.

/// The `chat_template` of `HuggingFaceTB/SmolVLM-256M-Instruct`'s `tokenizer_config.json`, verbatim.
pub const CHAT_TEMPLATE: &str = "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}";
