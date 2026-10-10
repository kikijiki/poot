//! The `poot-generate` binary at the process boundary: what its command line accepts and refuses. These rows
//! run the built binary, so a flag the parser lost or grew shows up here and not only in the library rows that
//! sit under it (`driver::oneshot`). None needs a model or a device: each ends in the argument parser or in
//! the backend open, both before a checkpoint is read.

use std::process::{Command, Output};

use poot_test_util::contained_child::ContainChild;

/// `CARGO_BIN_EXE_poot-generate`, read at runtime: cargo sets it only for test builds that enable the
/// package's `cli` feature, which the package's own dev-dependency does.
fn poot_generate(args: &[&str]) -> Output {
    let bin = std::env::var("CARGO_BIN_EXE_poot-generate")
        .expect("CARGO_BIN_EXE_poot-generate must be set by cargo for test binaries");
    Command::new(bin)
        .contained()
        .args(args)
        .output()
        .expect("spawn poot-generate")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A flag poot has no capability for is refused by name, and a positional argument (the old command
/// line) is refused too. Mutations: declare `--n-gpu-layers` (an offload poot cannot honour) on the
/// parser; accept positional arguments.
#[test]
fn a_flag_poot_lacks_and_a_positional_argument_are_refused() {
    for lacking in [
        &["-p", "hi", "--n-gpu-layers", "99"][..],
        &["/models/qwen", "The capital of France is", "16"][..],
    ] {
        let output = poot_generate(lacking);
        assert!(!output.status.success(), "{lacking:?} was accepted");
        assert!(
            stderr(&output).contains("unexpected argument"),
            "{lacking:?}: {}",
            stderr(&output)
        );
    }
}

/// A run with neither `-p` nor `-f` is refused with the missing flags named. Mutation: drop the
/// required input group.
#[test]
fn a_run_without_a_prompt_is_refused() {
    let output = poot_generate(&["-m", "/nonexistent"]);
    assert!(!output.status.success());
    let message = stderr(&output);
    assert!(
        message.contains("--prompt") && message.contains("--file"),
        "{message}"
    );
}

/// A conversation reads its turns from stdin, so it needs no `-p`/`-f`: the parser accepts it and the run
/// reaches the model load (the path does not exist), as `--single-turn` and `-p` with a conversation do.
/// Mutations: leave `-p`/`-f` required beside `--conversation`; drop `single_turn` from the
/// `required_unless_present_any` of `-p`.
#[test]
fn a_conversation_needs_no_prompt_flag() {
    for flags in [
        &["--conversation"][..],
        &["--single-turn"][..],
        &["--conversation", "-p", "hi"][..],
        &["--single-turn", "-p", "hi"][..],
    ] {
        let mut args = vec!["-m", "/nonexistent", "--backend", "wgpu"];
        args.extend_from_slice(flags);
        let output = poot_generate(&args);
        assert!(!output.status.success(), "{flags:?}");
        let message = stderr(&output);
        assert!(
            !message.contains("required arguments"),
            "{flags:?} was refused by the parser: {message}"
        );
    }
}

/// A conversation answers through the chat template, so asking for the prompt as written beside it is
/// refused naming both flags, before a model is read. Mutation: drop `conflicts_with_all` from
/// `--no-chat-template`.
#[test]
fn a_conversation_refuses_the_prompt_as_written() {
    for conversing in ["--conversation", "--single-turn"] {
        let output = poot_generate(&["-m", "/nonexistent", conversing, "--no-chat-template"]);
        assert!(!output.status.success());
        let message = stderr(&output);
        assert!(
            message.contains(conversing) && message.contains("--no-chat-template"),
            "{message}"
        );
    }
}

/// A chat template override is one template: given as text or as a file, not both, and not beside the
/// prompt as written. Each pair is refused naming both flags, before a model is read. Mutations: drop
/// `conflicts_with_all` from `--chat-template`; drop `conflicts_with` from `--chat-template-file`.
#[test]
fn a_chat_template_override_refuses_a_second_source_and_the_prompt_as_written() {
    for pair in [
        ["--chat-template", "--chat-template-file"],
        ["--chat-template", "--no-chat-template"],
        ["--chat-template-file", "--no-chat-template"],
    ] {
        let mut args = vec!["-m", "/nonexistent", "-p", "hi"];
        for flag in pair {
            args.push(flag);
            if flag != "--no-chat-template" {
                args.push("x");
            }
        }
        let output = poot_generate(&args);
        assert!(!output.status.success());
        let message = stderr(&output);
        assert!(
            message.contains("cannot be used with")
                && pair.iter().all(|flag| message.contains(flag)),
            "{pair:?}: {message}"
        );
    }
}

/// `--help` lists every flag the tool accepts and none it lacks. Mutations: rename a flag (`--ignore-eos`
/// to `--ignore_eos`); declare `--n-gpu-layers`.
#[test]
fn help_lists_the_accepted_flags_and_not_the_missing_ones() {
    let output = poot_generate(&["--help"]);
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    let words: Vec<&str> = help
        .split(|c: char| c.is_whitespace() || c == ',')
        .collect();
    for flag in [
        "--model",
        "--prompt",
        "--file",
        "--system-prompt",
        "--no-chat-template",
        "--chat-template",
        "--chat-template-file",
        "--conversation",
        "--single-turn",
        "--predict",
        "--ctx-size",
        "--temp",
        "--top-k",
        "--top-p",
        "--min-p",
        "--repeat-penalty",
        "--seed",
        "--reverse-prompt",
        "--ignore-eos",
        "--backend",
        "--show-timings",
        "--no-show-timings",
        "--verbose",
    ] {
        assert!(words.contains(&flag), "--help lacks {flag}:\n{help}");
    }
    assert!(
        !words.contains(&"--n-gpu-layers"),
        "--help lists --n-gpu-layers"
    );
}

/// A backend this build lacks is refused naming the cargo feature that enables it, before any model is
/// read (the model path does not exist). Mutation: drop the feature from `Unsupported::Backend`'s message.
#[cfg(not(feature = "rocm"))]
#[test]
fn a_backend_not_built_in_names_the_feature_to_enable() {
    let output = poot_generate(&["-m", "/nonexistent", "-p", "hi", "--backend", "rocm"]);
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("--features rocm"),
        "{}",
        stderr(&output)
    );
}
