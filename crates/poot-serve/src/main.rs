//! poot-serve: a minimal OpenAI-compatible HTTP server over the poot driver's [`poot_llm::driver::ModelHandle`].
//!
//! The server loads a checkpoint through the family registry, prepares it on the device its engine plan chose
//! and serves it from one scheduling loop over the driver's `step` (`batch`), on an engine thread that owns
//! the driver. Every route that runs the model sends that thread a message on one queue: generation
//! (completions and chat, `n` choices as `n` concurrent jobs), `Encode` steps (embeddings and rerank) and
//! LoRA administration. Pure std HTTP, serde JSON. Accept/parse/shape run on per-connection threads.
//!
//! Usage: `poot-serve [--backend wgpu|rocm|ptx|vulkan|auto] [MODEL_DIR] [ADDR]` (defaults: `auto`, the local
//! Qwen2.5-0.5B dir, `127.0.0.1:8080`). Startup computes one typed engine plan (`engine_plan`): it opens the
//! chosen device once, fits the workload to its memory and prepares it; `GET /health` is `503` until then. Logs go through `tracing` (`RUST_LOG`, default `info`). SIGINT/SIGTERM stop accepting
//! connections and let in-flight requests drain.
//!
//! Endpoints: `POST /v1/completions` and `POST /v1/chat/completions` (parsed, bounded, then run by the loop);
//! `POST /v1/embeddings` and `POST /v1/rerank` (hidden states pooled on the host, or a cross-encoder's
//! scores); `GET /v1/models`; `GET /health`; `GET /metrics` (JSON); `GET /metrics/prometheus`.
//! Sampling knobs are optional and default to greedy; `max_tokens` is bounded once at parse time.
//!
//! LoRA administration (`/v1/lora_adapters`) keeps its trust boundary: it is local-only on the default
//! loopback listener; on a non-loopback listener it is disabled unless `POOT_LORA_ADMIN_KEY` is set, in
//! which case every `/v1/lora_adapters` request needs `Authorization: Bearer <key>`. A request naming a
//! `lora_adapter` that no loaded adapter has is a 400. Ordinary inference endpoints are not authenticated.

// `deliver_terminal` publishes the request-outcome count before it sends the terminal event, so it has
// to know whether that event will reach a live receiver first; stable `mpsc::Sender` offers no
// liveness query. The workspace toolchain is the pinned nightly (`rust-toolchain.toml`).
#![feature(mpsc_is_disconnected)]
// `Metrics::json` is one `json!` literal of every gauge; its expansion recurses once per key.
#![recursion_limit = "256"]

mod admin;

mod api;

mod engine_plan;

mod batch;

mod handlers;

mod http;

mod lifecycle;

mod metrics;

#[cfg(test)]
mod tests;

mod types;

mod vision;

mod startup;

fn main() -> anyhow::Result<()> {
    startup::run()
}
