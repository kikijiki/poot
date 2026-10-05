//! poot-serve: a minimal OpenAI-compatible HTTP server over the poot [`poot_llm::Runner`].
//!
//! The server has no generation engine: it starts, parses and bounds requests, and
//! refuses a valid generation request with a typed `503` that names the missing engine; `GET /health`
//! reports `"engine_loaded": false`. Serving returns as one scheduling loop over the driver. Pure std
//! HTTP, serde JSON. Accept/parse/shape run on per-connection threads.
//!
//! Usage: `poot-serve [MODEL_DIR] [ADDR] [--lora-adapter NAME=DIR]... [--lora-pool-capacity N]` (defaults:
//! the local Qwen2.5-0.5B dir, `127.0.0.1:8080`). Logs go through `tracing` (`RUST_LOG`, default `info`).
//! SIGINT/SIGTERM stop accepting connections and let in-flight requests drain.
//!
//! Endpoints: `POST /v1/completions` and `POST /v1/chat/completions` (parsed, bounded, then `503`);
//! `POST /v1/embeddings` and `POST /v1/rerank` (decoder pooling, BERT encoder, or cross-encoder, on the
//! connection thread); `GET /v1/models`; `GET /health`; `GET /metrics` (JSON); `GET /metrics/prometheus`.
//! Sampling knobs are optional and default to greedy; `max_tokens` is bounded once at
//! parse time.
//!
//! LoRA adapters: repeatable `--lora-adapter NAME=DIR` flags register named PEFT adapter directories
//! (`adapter_config.json` + `adapter_model.safetensors`, `q_proj`/`k_proj`/`v_proj`/`o_proj`/
//! `gate_proj`/`up_proj`/`down_proj` targets) into the `Runner`'s
//! multi-adapter pool at startup. A request selects one by name via the `lora_adapter` field on
//! `/v1/completions`/`/v1/chat/completions` (absent or `null` = no adapter); an unknown name is a 400.
//! LoRA administration is local-only on the default loopback listener; on a non-loopback listener it is
//! disabled unless `POOT_LORA_ADMIN_KEY` is set, in which case every `/v1/lora_adapters` request needs
//! `Authorization: Bearer <key>`. Ordinary inference endpoints are not authenticated.

// `deliver_terminal` publishes the request-outcome count before it sends the terminal event, so it has
// to know whether that event will reach a live receiver first; stable `mpsc::Sender` offers no
// liveness query. The workspace toolchain is the pinned nightly (`rust-toolchain.toml`).
#![cfg_attr(test, feature(mpsc_is_disconnected))]

mod admin;

mod api;

// The scheduling pieces have no non-test caller until the one serving loop lands, so the module builds
// only for its own tests.
#[cfg(test)]
mod batch;

mod handlers;

mod http;

mod lifecycle;

mod metrics;

#[cfg(test)]
mod tests;

mod types;

mod startup;

fn main() -> anyhow::Result<()> {
    startup::run()
}

#[cfg(test)]
mod main_tests {
    use crate::startup::{parse_lora_adapter_flag, parse_lora_pool_capacity_flag};

    #[test]
    fn parse_lora_adapter_flag_splits_on_first_equals() {
        let (name, dir) = parse_lora_adapter_flag("mystyle=/adapters/mystyle").unwrap();
        assert_eq!(name, "mystyle");
        assert_eq!(dir, "/adapters/mystyle");
    }

    #[test]
    fn parse_lora_adapter_flag_keeps_a_later_equals_in_the_dir() {
        // A directory path can itself contain '=' (unusual but valid on Linux); only the FIRST '=' is
        // the name/dir separator.
        let (name, dir) = parse_lora_adapter_flag("a=b/c=d").unwrap();
        assert_eq!(name, "a");
        assert_eq!(dir, "b/c=d");
    }

    #[test]
    fn parse_lora_adapter_flag_rejects_missing_equals() {
        assert!(parse_lora_adapter_flag("no-equals-sign").is_err());
    }

    #[test]
    fn parse_lora_adapter_flag_rejects_empty_name_or_dir() {
        assert!(parse_lora_adapter_flag("=dir").is_err());
        assert!(parse_lora_adapter_flag("name=").is_err());
    }

    fn args(v: &[&str]) -> Vec<String> {
        std::iter::once("poot-serve".to_string())
            .chain(v.iter().map(|s| s.to_string()))
            .collect()
    }

    #[test]
    fn parse_lora_pool_capacity_flag_absent_is_none() {
        assert_eq!(
            parse_lora_pool_capacity_flag(&args(&["model", "addr"])).unwrap(),
            None
        );
    }

    #[test]
    fn parse_lora_pool_capacity_flag_parses_the_value() {
        assert_eq!(
            parse_lora_pool_capacity_flag(&args(&["--lora-pool-capacity", "4"])).unwrap(),
            Some(4)
        );
        // 0 is a legal (if pointless - no headroom beyond the --lora-adapter flags) value.
        assert_eq!(
            parse_lora_pool_capacity_flag(&args(&["--lora-pool-capacity", "0"])).unwrap(),
            Some(0)
        );
    }

    #[test]
    fn parse_lora_pool_capacity_flag_rejects_non_integer() {
        assert!(parse_lora_pool_capacity_flag(&args(&["--lora-pool-capacity", "two"])).is_err());
        assert!(parse_lora_pool_capacity_flag(&args(&["--lora-pool-capacity", "-1"])).is_err());
    }
}
