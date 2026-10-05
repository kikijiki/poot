//! The fallback chat-prompt format: a hardcoded template family the checkpoint's jinja
//! `chat_template` (tokenizer data) overrides when present. `ModelConfig::chat` carries it.

/// The chat-prompt template family for a model. The OpenAI `/v1/chat/completions` path renders the
/// role/content messages into the model's own template (feeding ChatML markup to a non-ChatML model degrades
/// output and its turn-end stop never matches). These hardcoded families are the fallback for a model with no
/// jinja `chat_template`; otherwise the runner renders that template. Each family names its own in its
/// `ModelConfig`: there is no lookup from an architecture string, so a family cannot inherit another's:
///
/// ```compile_fail,E0599
/// let _ = poot_models::chat::ChatFormat::for_arch("qwen2");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    /// Qwen2/Qwen3 (and the default): `<|im_start|>role\n…<|im_end|>`.
    ChatML,
    /// Phi-3/Phi-4: `<|role|>\n…<|end|>`.
    Phi3,
    /// Llama 3.x: `<|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>`.
    Llama3,
    /// Gemma: `<start_of_turn>role\n…<end_of_turn>` (assistant->model, system folded into user).
    Gemma,
    /// OLMo-2 (Tulu family): `<|user|>\n...\n<|assistant|>\n`, bos `<|endoftext|>`, assistant turns end with `<|endoftext|>`.
    Olmo2,
    /// IBM Granite (dense + granitemoe): `<|start_of_role|>role<|end_of_role|>...<|end_of_text|>\n`.
    Granite,
    /// Mistral: `[INST] {user} [/INST]{assistant}</s>` (no dedicated system role, like Gemma: a leading system
    /// message folds into the following user turn). A Mistral checkpoint with no jinja `chat_template` must not
    /// fall back to ChatML: `<|im_start|>`/`<|im_end|>` are not special tokens in the Mistral vocab, so they
    /// tokenize as literal text and degrade instruction-following.
    Mistral,
}

impl ChatFormat {
    /// The marker that ends an assistant turn: the stop sequence the server halts generation at.
    pub fn turn_end(&self) -> &'static str {
        match self {
            ChatFormat::ChatML => "<|im_end|>",
            ChatFormat::Phi3 => "<|end|>",
            ChatFormat::Llama3 => "<|eot_id|>",
            ChatFormat::Gemma => "<end_of_turn>",
            ChatFormat::Olmo2 => "<|endoftext|>",
            ChatFormat::Granite => "<|end_of_text|>",
            ChatFormat::Mistral => "</s>",
        }
    }

    /// Render `(role, content)` messages into the model's prompt, ending with the assistant turn opener so
    /// generation continues the assistant message. Pure (no model state).
    pub fn render(&self, messages: &[(&str, &str)]) -> String {
        let mut s = String::new();
        match self {
            ChatFormat::ChatML => {
                for (role, content) in messages {
                    s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
                }
                s.push_str("<|im_start|>assistant\n");
            }
            ChatFormat::Phi3 => {
                for (role, content) in messages {
                    s.push_str(&format!("<|{role}|>\n{content}<|end|>\n"));
                }
                s.push_str("<|assistant|>\n");
            }
            ChatFormat::Llama3 => {
                s.push_str("<|begin_of_text|>");
                for (role, content) in messages {
                    s.push_str(&format!(
                        "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
                    ));
                }
                s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            ChatFormat::Gemma => {
                // Gemma has only user/model turns and is trained on strict user/model alternation, so a leading system
                // message must be folded into the following user turn: a separate `<start_of_turn>user` for system followed
                // by another for the user turn is off-distribution. Map assistant->model, system+user->user, then merge
                // consecutive same-role turns (content joined with a blank line).
                let mut merged: Vec<(&str, String)> = Vec::new();
                for (role, content) in messages {
                    let r = match *role {
                        "assistant" | "model" => "model",
                        _ => "user", // system + user
                    };
                    match merged.last_mut() {
                        Some(last) if last.0 == r => {
                            last.1.push_str("\n\n");
                            last.1.push_str(content);
                        }
                        _ => merged.push((r, (*content).to_string())),
                    }
                }
                for (r, content) in &merged {
                    s.push_str(&format!("<start_of_turn>{r}\n{content}<end_of_turn>\n"));
                }
                s.push_str("<start_of_turn>model\n");
            }
            ChatFormat::Olmo2 => {
                s.push_str("<|endoftext|>");
                for (role, content) in messages {
                    match *role {
                        "assistant" => {
                            s.push_str(&format!("<|assistant|>\n{content}<|endoftext|>\n"))
                        }
                        _ => s.push_str(&format!("<|{role}|>\n{content}\n")),
                    }
                }
                s.push_str("<|assistant|>\n");
            }
            ChatFormat::Granite => {
                for (role, content) in messages {
                    s.push_str(&format!(
                        "<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n"
                    ));
                }
                s.push_str("<|start_of_role|>assistant<|end_of_role|>");
            }
            ChatFormat::Mistral => {
                // Mistral has no dedicated system role (the official template errors on one); fold a leading system message
                // into the following user turn's `[INST]` block, as Gemma does above.
                s.push_str("<s>");
                let mut pending_system: Option<&str> = None;
                for (role, content) in messages {
                    match *role {
                        "system" => pending_system = Some(content),
                        "assistant" => {
                            s.push_str(content);
                            s.push_str("</s>");
                        }
                        _ => {
                            s.push_str("[INST] ");
                            if let Some(sys) = pending_system.take() {
                                s.push_str(sys);
                                s.push_str("\n\n");
                            }
                            s.push_str(content);
                            s.push_str(" [/INST]");
                        }
                    }
                }
            }
        }
        s
    }
}

#[cfg(test)]
mod chat_format {
    use super::ChatFormat;

    const MSGS: &[(&str, &str)] = &[("system", "Be terse."), ("user", "Hi")];

    #[test]
    fn chatml_matches_the_prior_hardcoded_template() {
        // Byte-identical to poot-serve's old hardcoded ChatML (role concat + assistant opener).
        let got = ChatFormat::ChatML.render(MSGS);
        assert_eq!(
            got,
            "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(ChatFormat::ChatML.turn_end(), "<|im_end|>");
    }

    #[test]
    fn phi3_template() {
        assert_eq!(
            ChatFormat::Phi3.render(MSGS),
            "<|system|>\nBe terse.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\n"
        );
        assert_eq!(ChatFormat::Phi3.turn_end(), "<|end|>");
    }

    #[test]
    fn llama3_template() {
        assert_eq!(
            ChatFormat::Llama3.render(MSGS),
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe terse.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert_eq!(ChatFormat::Llama3.turn_end(), "<|eot_id|>");
    }

    #[test]
    fn gemma_folds_system_into_user_and_maps_model() {
        // System is folded into the following user turn (joined by a blank line): Gemma requires strict user/model
        // alternation, so a separate `<start_of_turn>user` for system would be a second consecutive user turn
        // (off-distribution). assistant maps to model.
        assert_eq!(
            ChatFormat::Gemma.render(&[("system", "S"), ("user", "U"), ("assistant", "A")]),
            "<start_of_turn>user\nS\n\nU<end_of_turn>\n\
             <start_of_turn>model\nA<end_of_turn>\n<start_of_turn>model\n"
        );
        assert_eq!(ChatFormat::Gemma.turn_end(), "<end_of_turn>");
    }

    #[test]
    fn gemma_merges_consecutive_same_role_turns() {
        // A lone user turn is untouched.
        assert_eq!(
            ChatFormat::Gemma.render(&[("user", "U")]),
            "<start_of_turn>user\nU<end_of_turn>\n<start_of_turn>model\n"
        );
        // Two consecutive user turns (post role-map) merge into one, never two user blocks in a row.
        assert_eq!(
            ChatFormat::Gemma.render(&[("system", "S"), ("user", "U")]),
            "<start_of_turn>user\nS\n\nU<end_of_turn>\n<start_of_turn>model\n"
        );
        // The rendered prompt must never contain two consecutive user turns.
        let out = ChatFormat::Gemma.render(&[("system", "S1"), ("system", "S2"), ("user", "U")]);
        assert!(
            !out.contains("<end_of_turn>\n<start_of_turn>user"),
            "consecutive user turns in Gemma render: {out}"
        );
        assert_eq!(
            out,
            "<start_of_turn>user\nS1\n\nS2\n\nU<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn olmo2_template() {
        assert_eq!(
            ChatFormat::Olmo2.render(&[("system", "S"), ("user", "U"), ("assistant", "A")]),
            "<|endoftext|><|system|>\nS\n<|user|>\nU\n<|assistant|>\nA<|endoftext|>\n<|assistant|>\n"
        );
        assert_eq!(
            ChatFormat::Olmo2.render(&[("user", "U")]),
            "<|endoftext|><|user|>\nU\n<|assistant|>\n"
        );
        assert_eq!(ChatFormat::Olmo2.turn_end(), "<|endoftext|>");
    }

    #[test]
    fn granite_template() {
        assert_eq!(
            ChatFormat::Granite.render(&[("system", "S"), ("user", "U"), ("assistant", "A")]),
            "<|start_of_role|>system<|end_of_role|>S<|end_of_text|>\n\
             <|start_of_role|>user<|end_of_role|>U<|end_of_text|>\n\
             <|start_of_role|>assistant<|end_of_role|>A<|end_of_text|>\n\
             <|start_of_role|>assistant<|end_of_role|>"
        );
        assert_eq!(
            ChatFormat::Granite.render(&[("user", "U")]),
            "<|start_of_role|>user<|end_of_role|>U<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        );
        assert_eq!(ChatFormat::Granite.turn_end(), "<|end_of_text|>");
    }

    #[test]
    fn mistral_template() {
        // System has no dedicated role (the official template errors on one); it folds into the following user turn's [INST] block, as in Gemma.
        assert_eq!(
            ChatFormat::Mistral.render(&[("system", "S"), ("user", "U"), ("assistant", "A")]),
            "<s>[INST] S\n\nU [/INST]A</s>"
        );
        assert_eq!(
            ChatFormat::Mistral.render(&[("user", "U")]),
            "<s>[INST] U [/INST]"
        );
        // A second turn continues after the first assistant reply, no system to fold this time.
        assert_eq!(
            ChatFormat::Mistral.render(&[("user", "U1"), ("assistant", "A1"), ("user", "U2"),]),
            "<s>[INST] U1 [/INST]A1</s>[INST] U2 [/INST]"
        );
        assert_eq!(ChatFormat::Mistral.turn_end(), "</s>");
    }
}
