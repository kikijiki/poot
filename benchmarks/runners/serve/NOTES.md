# serve-throughput runner notes (POOT-1334)

The scenario measures an engine's server under concurrent streaming requests. `client.py` is the one load client (stdlib
only) and `run.py` starts each engine's server for one cell, drives it and prints one JSON line. The harness validates and
aggregates it (`harness/result_contract.py`, `_validate_serve_result`).

## Prior art, and what this takes from it

Sources read for this card (2026-11-16):

- **vLLM `benchmark_serving`** (`vllm/benchmarks/serve.py`,
  <https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py>). Arrival is `--request-rate` (default `inf`: all
  requests at time 0; otherwise Poisson, or gamma with `--burstiness`), and `--max-concurrency` caps the requests in flight with a
  semaphore, so the actual rate can be lower than asked. Metrics: TTFT per request, TPOT as `(latency - ttft) / (output_len - 1)`,
  ITL over every inter-token gap, E2EL, output throughput as `sum(output lens) / duration`, and goodput (`--goodput ttft:.. tpot:..
e2el:..`: requests meeting every SLO per second). Output token counts come from the server's usage when present.
  _Taken:_ the TPOT and output-throughput definitions and the server-reported token count. _Different:_ the load here is
  closed loop at a fixed concurrency, which is vLLM's `--max-concurrency` with `--request-rate inf`; the concurrency is the
  independent variable the card names, and a closed loop cannot overload the server into a queue whose length is the result.
  Goodput is not stated: it needs an SLO the scenario does not own.
- **MLPerf Inference LoadGen** (<https://github.com/mlcommons/inference/blob/master/loadgen/test_settings.h>). Its Server
  scenario sends queries with Poisson arrivals at a target QPS and passes when a latency percentile is under a bound; SingleStream
  sends the next query when the previous completes; Offline sends everything at once and reports samples per second. _Taken:_
  nothing of the arrival model. A Poisson server scenario answers "how much load meets this latency bound", which needs the
  bound; this scenario answers "what does the engine deliver at N requests in flight", the question of the card, and its
  concurrency 1 is MLPerf's SingleStream.
- **llama.cpp**. `llama-bench` (<https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md>) has no server or
  concurrency mode (prompt processing `-p`, generation `-n`, batch sizes, repeats `-r`) and excludes tokenization and sampling.
  The concurrent measure is `llama-server` (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>):
  `-np/--parallel` slots, continuous batching on by default, one context shared by the slots (`-c`), `cache_prompt`
  (reuse of a previous request's KV) and `ignore_eos` on `/completion`. _Taken:_ the server, with `--parallel` = concurrency,
  `-c` = slots x positions, and `cache_prompt: false` in every request.

## Decisions

- **One client for all engines**, so TTFT/TPOT are measured one way. The engines' own metrics (poot's receipt chunk, vLLM's
  stats, llama-server's `timings`) are not read.
- **A prompt of exact length without a tokenizer.** Prompts are words drawn from a pool of single-token words; the client asks the
  server for the token count of a probe prompt and adds or removes words until it is the scenario's length (this absorbs a BOS
  or a template), then every request must report that count.
- **Fixed output length** by `ignore_eos` (vLLM's field; llama-server's `/completion` field of the same name; poot-serve
  accepts it on completions and chat since POOT-1334). A request that ends early reports fewer tokens and fails the cell.
- **Distinct prompts, prefix caching off** where the server has the switch, so a prefix cache cannot turn the prefill of
  later requests into a lookup.
- **A server per cell**, started on a free loopback port with exactly `concurrency` slots and positions for one request each,
  and stopped (its whole process group) after the cell. The observer's memory is that server's.
- **`server_slots`** is what the server states (poot: `slot_capacity` of `/metrics`, after its memory fit; the others: the flag
  they were started with). A server that fits fewer rows than the concurrency fails the cell.

## Per engine

- **poot**: `poot-serve MODEL 127.0.0.1:PORT --backend B --rows C --max-seq S`. Identity from `GET /v1/identity`. The backend is
  the one the server was started on (it fails startup rather than open another).
- **vLLM**: `python -m vllm.entrypoints.openai.api_server` with `--max-num-seqs C --max-model-len S
--no-enable-prefix-caching --gpu-memory-utilization 0.85`; version from `GET /version`. Its KV pool inflates external VRAM
  (a row caveat, as for the single-stream runner). Not exercised on a pod yet.
- **llama.cpp**: `llama-server -m GGUF -ngl 99 --parallel C -c C*S`; build from `GET /props`; the backend from the server's own
  start-up log (a run on the CPU is refused). The GGUF is resolved (and converted) as for the single-stream runner. Not
  exercised on a pod yet.
- **candle, transformers**: no server that batches concurrent requests; `unsupported` with that reason (the scenario's
  `unsupported` table in `manifest.toml`).
