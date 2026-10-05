//! Server metrics: latency histograms, counters, gauges, and their JSON/Prometheus rendering.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Upper bucket bounds (`le`, ms) for the latency histograms; the implicit `+Inf` bucket catches the rest.
pub(crate) const LATENCY_BUCKETS_MS: [f64; 11] = [
    5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
];

/// Lock-free Prometheus-style latency histogram over [`LATENCY_BUCKETS_MS`]: non-cumulative per-bucket counts
/// plus running sum and count. The `+Inf` bucket is `count - sum(buckets)`.
pub(crate) struct Histogram {
    /// one counter per bound in `LATENCY_BUCKETS_MS` (observations that fall in `(prev, le]`).
    pub(crate) buckets: [AtomicU64; LATENCY_BUCKETS_MS.len()],
    /// observations above the last bound.
    pub(crate) overflow: AtomicU64,
    /// sum of observed values in microseconds (integer so it is atomic; rendered back to ms).
    pub(crate) sum_us: AtomicU64,
    pub(crate) count: AtomicU64,
}

impl Histogram {
    pub(crate) fn new() -> Self {
        Histogram {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            overflow: AtomicU64::new(0),
            sum_us: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record one latency observation in milliseconds.
    pub(crate) fn observe(&self, ms: f64) {
        let ms = ms.max(0.0);
        match LATENCY_BUCKETS_MS.iter().position(|&le| ms <= le) {
            Some(i) => self.buckets[i].fetch_add(1, Ordering::Relaxed),
            None => self.overflow.fetch_add(1, Ordering::Relaxed),
        };
        self.sum_us
            .fetch_add((ms * 1000.0) as u64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub(crate) fn sum_ms(&self) -> f64 {
        self.sum_us.load(Ordering::Relaxed) as f64 / 1000.0
    }

    pub(crate) fn avg_ms(&self) -> f64 {
        let c = self.count();
        if c == 0 {
            0.0
        } else {
            self.sum_ms() / c as f64
        }
    }

    /// The Prometheus histogram exposition for this metric: cumulative `_bucket{le=...}` series (monotonic
    /// in `le`, with a `+Inf` bucket equal to the total count), then `_sum` and `_count`.
    pub(crate) fn prometheus(&self, name: &str, help: &str) -> String {
        let mut out = format!("# HELP {name} {help}\n# TYPE {name} histogram\n");
        let mut cumulative = 0u64;
        for (i, &le) in LATENCY_BUCKETS_MS.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            out.push_str(&format!("{name}_bucket{{le=\"{le}\"}} {cumulative}\n"));
        }
        cumulative += self.overflow.load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}\n"));
        out.push_str(&format!("{name}_sum {:.3}\n", self.sum_ms()));
        out.push_str(&format!("{name}_count {}\n", self.count()));
        out
    }
}

/// Paged-KV, prefix-cache and GPU-memory gauges. The batched engine updates them as it admits and steps; the
/// CPU path leaves them 0.
#[derive(Default)]
pub(crate) struct CacheMetrics {
    /// physical KV blocks in use across all slots (stored each step).
    pub(crate) kv_used_blocks: AtomicU64,
    /// total physical KV blocks in the shared pool (set once at engine start; 0 on the CPU path).
    pub(crate) kv_total_blocks: AtomicU64,
    /// cumulative prompt tokens served from a cached prefix - the prefix-cache hit-rate numerator.
    pub(crate) prefix_hit_tokens: AtomicU64,
    /// cumulative prompt tokens looked up against the prefix cache - the hit-rate denominator.
    pub(crate) prefix_query_tokens: AtomicU64,
    /// model weight bytes resident in GPU memory (set once at engine start; the fixed VRAM cost).
    pub(crate) gpu_weight_bytes: AtomicU64,
    /// KV-pool bytes resident in GPU memory (set once at engine start; the shared paged pool's buffers).
    pub(crate) gpu_kv_bytes: AtomicU64,
    /// cumulative ns inside the GPU decode step (the `batched_decode_step_paged` call): numerator of the
    /// GPU-active ratio. Timed by poot itself, no vendor API; low means host/launch-bound, high means GPU-bound.
    pub(crate) decode_gpu_ns: AtomicU64,
    /// cumulative ns of the whole decode step (GPU call plus host slot bookkeeping/sampling): the denominator.
    pub(crate) decode_step_ns: AtomicU64,
    /// device-wide VRAM in use (bytes), refreshed each decode step from the backend's runtime API: Vulkan
    /// `VK_EXT_memory_budget` (`GpuExecutor::vram_used_bytes`) on wgpu, CUDA `cuMemGetInfo`
    /// (`PtxContext::vram_used_bytes`, Card 549: moved here with the rest of the executor contract,
    /// `PtxGraphExecutor`'s own copy deleted) on PTX, HSA `HSA_AMD_AGENT_INFO_MEMORY_AVAIL`
    /// (`RocmContext::vram_used_bytes`, Card 548: moved here with the rest of the executor contract's
    /// memory service, `RocmGraphExecutor`'s own copy deleted) on ROCm. Includes co-resident
    /// processes. 0 until the first step or if the query is unavailable.
    pub(crate) gpu_vram_used_bytes: AtomicU64,
    /// total device VRAM visible to this process (bytes): Vulkan `VK_EXT_memory_budget` `heap_budget`
    /// (`GpuExecutor::vram_budget_bytes`) on wgpu, CUDA `cuMemGetInfo` total (`PtxContext::vram_budget_bytes`,
    /// Card 549: moved here with the rest of the executor contract, `PtxGraphExecutor`'s own copy
    /// deleted) on PTX, HSA `HSA_AMD_MEMORY_POOL_INFO_SIZE` (`RocmContext::vram_budget_bytes`, Card 548: moved
    /// here with the rest of the executor contract's memory service, `RocmGraphExecutor`'s own copy
    /// deleted) on ROCm. Refreshed per decode step; 0 if the query is unavailable.
    pub(crate) gpu_vram_total_bytes: AtomicU64,
    /// per-token GPU dispatch count of the optimized decode graph (set once at engine start): the graph eqn count
    /// after cse/flash/fuse, i.e. the kernel launches per token that capture/replay collapses to one host call.
    /// A qwen2-style estimate (see `Runner::decode_graph_cost`).
    pub(crate) decode_dispatch_count: AtomicU64,
    /// peak transient activation bytes of the optimized decode graph (set once at engine start), by a liveness
    /// sweep; the per-token working set apart from weights and KV.
    pub(crate) decode_activation_peak_bytes: AtomicU64,
    /// GPU dispatch count of the optimized prefill (TTFT) graph at the KV capacity (set once at engine start).
    pub(crate) prefill_dispatch_count: AtomicU64,
    /// peak transient activation bytes of the optimized prefill graph at the KV capacity (set once at engine
    /// start); flash attention keeps this O(L*head_dim), not O(L^2).
    pub(crate) prefill_activation_peak_bytes: AtomicU64,
}

/// Batched-engine speculative-decode telemetry, with one counter pair per sampler mode: greedy rows
/// (argmax prefix-compare) and sampled rows (`spec_sample_round`) have distinct acceptance dynamics. Counted at
/// the per-row branch point in `speculative_verify_step`/`_ptx`/`_rocm`.
#[derive(Default)]
pub(crate) struct SpeculativeMetrics {
    /// draft tokens forwarded to the batched verify step (before accept/reject) for greedy rows.
    pub(crate) forwarded_greedy: AtomicU64,
    /// tokens committed to a slot via the greedy speculative path: accepted draft prefix plus bonus token, counted
    /// by the actual push (a token discarded as EOS, or a round cut short by max_new/a stop sequence, is not
    /// counted; see `speculative_verify_step`).
    pub(crate) committed_greedy: AtomicU64,
    /// as `forwarded_greedy`, for sampled rows.
    pub(crate) forwarded_sampled: AtomicU64,
    /// as `committed_greedy`, for sampled rows.
    pub(crate) committed_sampled: AtomicU64,
}

pub(crate) struct Metrics {
    pub(crate) started: Instant,
    pub(crate) requests: AtomicU64,
    pub(crate) completion_tokens: AtomicU64,
    /// prefill tokens processed (the prefill-throughput numerator; decode is `completion_tokens`).
    pub(crate) prompt_tokens: AtomicU64,
    /// requests that finished normally: a natural stop condition or `max_new_tokens` reached
    /// (`GenEvent::Done`). vLLM's `finish_reason in {stop, length}`.
    pub(crate) requests_completed: AtomicU64,
    /// requests evicted because the client disconnected before completion (reply channel closed mid-stream).
    /// Unlike `requests_preempted`, a cancellation ends the request for good.
    pub(crate) requests_cancelled: AtomicU64,
    /// requests that ended in an engine/generation error (`GenEvent::Failed` or equivalent).
    pub(crate) requests_errored: AtomicU64,
    /// cumulative preemption events (recompute mode): a running slot's KV blocks were freed and the request
    /// requeued so another admission could use the pool space. Not terminal; the request resumes later, possibly
    /// recomputing part of its generated span. One request can be preempted repeatedly, so this can exceed
    /// `requests`. Mirrors vLLM's `num_preemptions_total`.
    pub(crate) requests_preempted: AtomicU64,
    /// generation requests currently being processed (a gauge: +1 on entry, -1 on completion/error).
    pub(crate) inflight: AtomicU64,
    /// time-to-first-token distribution (streaming requests, where TTFT is observable).
    pub(crate) ttft: Histogram,
    /// inter-token latency distribution (streaming requests: the gap between consecutive emitted tokens).
    pub(crate) itl: Histogram,
    /// end-to-end request-latency distribution (all generation requests).
    pub(crate) e2e: Histogram,
    /// queue-wait distribution (GPU batched path): time enqueued before a decode slot freed. Excludes prefill,
    /// unlike TTFT.
    pub(crate) queue_wait: Histogram,
    /// the most recent batched-decode width (active slots) on the GPU path; 0 on the CPU fallback.
    pub(crate) batch_size: AtomicU64,
    /// the batched-decode slot capacity (POOT_SLOTS on GPU, 1 on the sequential CPU fallback).
    pub(crate) slot_capacity: u64,
    pub(crate) on_gpu: bool,
    pub(crate) profile: bool,
    /// the served model's name (the model-dir basename), echoed in responses + `/v1/models`.
    pub(crate) model: String,
    /// persistent so CPU usage is a meaningful delta since the previous `/metrics` refresh.
    pub(crate) sys: Mutex<sysinfo::System>,
    /// paged-KV utilization + prefix-cache hit rate (GPU batched engine; 0 on the CPU path).
    pub(crate) cache: CacheMetrics,
    /// batched-engine speculative-decode counters; 0 unless `POOT_SPEC_DRAFT_MODEL`/`POOT_SPEC_LOOKUP_NGRAM` is
    /// configured and at least one row has drafted.
    pub(crate) speculative: SpeculativeMetrics,
}

/// RAII guard for the in-flight gauge: increments on construction, decrements on drop, so early returns and
/// errors are still uncounted.
pub(crate) struct InFlight<'a>(&'a Metrics);
impl<'a> InFlight<'a> {
    pub(crate) fn enter(m: &'a Metrics) -> Self {
        m.inflight.fetch_add(1, Ordering::Relaxed);
        InFlight(m)
    }
}
impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.inflight.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Metrics {
    /// Record a finished generation request: end-to-end latency, prefill token count and (streaming only) TTFT.
    /// The engine bumps the decode-token counter per token.
    pub(crate) fn record_request(&self, e2e_ms: f64, prompt_tokens: usize, ttft_ms: Option<f64>) {
        self.e2e.observe(e2e_ms);
        self.prompt_tokens
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        if let Some(t) = ttft_ms {
            self.ttft.observe(t);
        }
    }

    /// Terminal `GenEvent::Done` accounting: a delivered `Done` is a completion; an undelivered one
    /// (the reply receiver is already gone) means the client disconnected, so it counts as a
    /// cancellation. `engine_loop` passes the delivery outcome read from the channel *before* it
    /// sends, so the count exists before the event can be observed; the batched paths pass the send
    /// result.
    #[cfg(test)]
    pub(crate) fn record_terminal(&self, delivered: bool) {
        if delivered {
            self.requests_completed.fetch_add(1, Ordering::Relaxed);
        } else {
            self.requests_cancelled.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Terminal failure accounting: `requests_errored`, whether or not a receiver is left, published
    /// before the failure event exists - the rule [`crate::types::GenEvent::failed`] documents and
    /// every terminal failure routes through, so the increment lives in exactly one place.
    #[cfg(test)]
    pub(crate) fn record_terminal_failure(&self) {
        self.requests_errored.fetch_add(1, Ordering::Relaxed);
    }

    /// KV-cache block utilization in `[0, 1]` (used / total); 0 when no paged pool exists (CPU path).
    pub(crate) fn kv_cache_usage(&self) -> f64 {
        let total = self.cache.kv_total_blocks.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.cache.kv_used_blocks.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Cumulative prefix-cache hit rate in `[0, 1]` (prompt tokens served from cache / looked up); 0 before
    /// any request (no lookups yet).
    pub(crate) fn prefix_cache_hit_rate(&self) -> f64 {
        let q = self.cache.prefix_query_tokens.load(Ordering::Relaxed);
        if q == 0 {
            return 0.0;
        }
        self.cache.prefix_hit_tokens.load(Ordering::Relaxed) as f64 / q as f64
    }

    /// Resident GPU memory the engine tracks: model weights plus the KV pool. Excludes transient intermediates;
    /// 0 on the CPU path.
    pub(crate) fn gpu_resident_bytes(&self) -> u64 {
        self.cache.gpu_weight_bytes.load(Ordering::Relaxed)
            + self.cache.gpu_kv_bytes.load(Ordering::Relaxed)
    }

    /// Requests admitted but waiting: `inflight` minus `batch_size`, i.e. jobs in the engine channel because every
    /// slot is busy (vLLM's `num_requests_waiting`). `saturating_sub` so a brief gauge skew reads 0.
    pub(crate) fn queue_depth(&self) -> u64 {
        self.inflight
            .load(Ordering::Relaxed)
            .saturating_sub(self.batch_size.load(Ordering::Relaxed))
    }

    /// GPU-active ratio (0..1): cumulative time inside the GPU decode step over cumulative whole-step wall time.
    /// Low means host/launch-bound, high means GPU-bound. 0 before any step or on the CPU path. See
    /// docs/launch-overhead.md.
    pub(crate) fn gpu_active_ratio(&self) -> f64 {
        let gpu = self.cache.decode_gpu_ns.load(Ordering::Relaxed);
        let total = self.cache.decode_step_ns.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            (gpu as f64 / total as f64).min(1.0)
        }
    }

    pub(crate) fn json(&self) -> String {
        let mut sys = self.sys.lock().unwrap();
        sys.refresh_memory();
        sys.refresh_cpu_usage();
        let mb = |bytes: u64| bytes / (1024 * 1024);
        let round1 = |x: f64| (x * 10.0).round() / 10.0;
        let json = serde_json::json!({
            "uptime_secs": self.started.elapsed().as_secs(),
            "requests": self.requests.load(Ordering::Relaxed),
            "completion_tokens": self.completion_tokens.load(Ordering::Relaxed),
            "prompt_tokens": self.prompt_tokens.load(Ordering::Relaxed),
            "requests_completed": self.requests_completed.load(Ordering::Relaxed),
            "requests_cancelled": self.requests_cancelled.load(Ordering::Relaxed),
            "requests_errored": self.requests_errored.load(Ordering::Relaxed),
            "requests_preempted": self.requests_preempted.load(Ordering::Relaxed),
            "inflight_requests": self.inflight.load(Ordering::Relaxed),
            "queue_depth": self.queue_depth(),
            "batch_size": self.batch_size.load(Ordering::Relaxed),
            "slot_capacity": self.slot_capacity,
            "kv_cache_usage": round1(self.kv_cache_usage() * 100.0) / 100.0,
            "kv_cache_blocks_used": self.cache.kv_used_blocks.load(Ordering::Relaxed),
            "kv_cache_blocks_total": self.cache.kv_total_blocks.load(Ordering::Relaxed),
            "prefix_cache_hit_rate": round1(self.prefix_cache_hit_rate() * 100.0) / 100.0,
            "gpu_weight_bytes": self.cache.gpu_weight_bytes.load(Ordering::Relaxed),
            "gpu_kv_cache_bytes": self.cache.gpu_kv_bytes.load(Ordering::Relaxed),
            "gpu_resident_bytes": self.gpu_resident_bytes(),
            "gpu_vram_used_bytes": self.cache.gpu_vram_used_bytes.load(Ordering::Relaxed),
            "gpu_vram_total_bytes": self.cache.gpu_vram_total_bytes.load(Ordering::Relaxed),
            "decode_dispatch_count": self.cache.decode_dispatch_count.load(Ordering::Relaxed),
            "decode_activation_peak_bytes": self.cache.decode_activation_peak_bytes.load(Ordering::Relaxed),
            "prefill_dispatch_count": self.cache.prefill_dispatch_count.load(Ordering::Relaxed),
            "prefill_activation_peak_bytes": self.cache.prefill_activation_peak_bytes.load(Ordering::Relaxed),
            "gpu_active_ratio": round1(self.gpu_active_ratio() * 100.0) / 100.0,
            "speculative_forwarded_tokens_greedy": self.speculative.forwarded_greedy.load(Ordering::Relaxed),
            "speculative_committed_tokens_greedy": self.speculative.committed_greedy.load(Ordering::Relaxed),
            "speculative_forwarded_tokens_sampled": self.speculative.forwarded_sampled.load(Ordering::Relaxed),
            "speculative_committed_tokens_sampled": self.speculative.committed_sampled.load(Ordering::Relaxed),
            "ttft_ms": { "count": self.ttft.count(), "avg": round1(self.ttft.avg_ms()) },
            "itl_ms": { "count": self.itl.count(), "avg": round1(self.itl.avg_ms()) },
            "request_e2e_ms": { "count": self.e2e.count(), "avg": round1(self.e2e.avg_ms()) },
            "queue_wait_ms": { "count": self.queue_wait.count(), "avg": round1(self.queue_wait.avg_ms()) },
            "model": &self.model,
            "decode_path": if self.on_gpu { "gpu" } else { "cpu" },
            "profiling": self.profile,
            "cpu_percent": round1(sys.global_cpu_usage() as f64),
            "mem_used_mb": mb(sys.used_memory()),
            "mem_total_mb": mb(sys.total_memory()),
        });
        json.to_string()
    }

    /// Prometheus text-exposition rendering (`GET /metrics/prometheus`). Counters are `_total`; the decode path is a
    /// labelled gauge. The JSON form stays on `/metrics`.
    pub(crate) fn prometheus(&self) -> String {
        let mut sys = self.sys.lock().unwrap();
        sys.refresh_memory();
        sys.refresh_cpu_usage();
        let mb = |bytes: u64| bytes / (1024 * 1024);
        let cpu = ((sys.global_cpu_usage() as f64) * 10.0).round() / 10.0;
        format!(
            "# HELP poot_uptime_seconds Server uptime.\n\
             # TYPE poot_uptime_seconds gauge\n\
             poot_uptime_seconds {}\n\
             # HELP poot_requests_total Generation requests served.\n\
             # TYPE poot_requests_total counter\n\
             poot_requests_total {}\n\
             # HELP poot_completion_tokens_total Completion (decode) tokens generated.\n\
             # TYPE poot_completion_tokens_total counter\n\
             poot_completion_tokens_total {}\n\
             # HELP poot_prompt_tokens_total Prompt (prefill) tokens processed.\n\
             # TYPE poot_prompt_tokens_total counter\n\
             poot_prompt_tokens_total {}\n\
             # HELP poot_requests_completed_total Requests that finished normally (natural stop or max_new_tokens reached).\n\
             # TYPE poot_requests_completed_total counter\n\
             poot_requests_completed_total {}\n\
             # HELP poot_requests_cancelled_total Requests evicted because the client disconnected before completion.\n\
             # TYPE poot_requests_cancelled_total counter\n\
             poot_requests_cancelled_total {}\n\
             # HELP poot_requests_errored_total Requests that ended in an engine/generation error.\n\
             # TYPE poot_requests_errored_total counter\n\
             poot_requests_errored_total {}\n\
             # HELP poot_requests_preempted_total Preemption events (recompute mode): a running request's KV was freed and it was requeued, not terminated.\n\
             # TYPE poot_requests_preempted_total counter\n\
             poot_requests_preempted_total {}\n\
             # HELP poot_inflight_requests Generation requests currently being processed.\n\
             # TYPE poot_inflight_requests gauge\n\
             poot_inflight_requests {}\n\
             # HELP poot_queue_depth Requests waiting for a free decode slot (in-flight minus actively decoding); vLLM num_requests_waiting.\n\
             # TYPE poot_queue_depth gauge\n\
             poot_queue_depth {}\n\
             # HELP poot_batch_size Active batched-decode slots in the last engine step (GPU path).\n\
             # TYPE poot_batch_size gauge\n\
             poot_batch_size {}\n\
             # HELP poot_slot_capacity Batched-decode slot capacity.\n\
             # TYPE poot_slot_capacity gauge\n\
             poot_slot_capacity {}\n\
             # HELP poot_decode_path Decode backend in use (1 for the active path).\n\
             # TYPE poot_decode_path gauge\n\
             poot_decode_path{{path=\"{}\"}} 1\n\
             # HELP poot_cpu_percent System CPU utilization since the last scrape.\n\
             # TYPE poot_cpu_percent gauge\n\
             poot_cpu_percent {}\n\
             # HELP poot_mem_used_megabytes Used system memory.\n\
             # TYPE poot_mem_used_megabytes gauge\n\
             poot_mem_used_megabytes {}\n\
             # HELP poot_mem_total_megabytes Total system memory.\n\
             # TYPE poot_mem_total_megabytes gauge\n\
             poot_mem_total_megabytes {}\n\
             # HELP poot_kv_cache_usage_ratio Paged KV-cache block utilization (used/total), 0..1.\n\
             # TYPE poot_kv_cache_usage_ratio gauge\n\
             poot_kv_cache_usage_ratio {}\n\
             # HELP poot_kv_cache_blocks_used Paged KV-cache blocks currently in use.\n\
             # TYPE poot_kv_cache_blocks_used gauge\n\
             poot_kv_cache_blocks_used {}\n\
             # HELP poot_kv_cache_blocks_total Paged KV-cache blocks in the shared pool.\n\
             # TYPE poot_kv_cache_blocks_total gauge\n\
             poot_kv_cache_blocks_total {}\n\
             # HELP poot_prefix_cache_hit_rate Cumulative prefix-cache hit rate (cached/looked-up prompt tokens), 0..1.\n\
             # TYPE poot_prefix_cache_hit_rate gauge\n\
             poot_prefix_cache_hit_rate {}\n\
             # HELP poot_prefix_cache_hit_tokens_total Prompt tokens served from a cached prefix.\n\
             # TYPE poot_prefix_cache_hit_tokens_total counter\n\
             poot_prefix_cache_hit_tokens_total {}\n\
             # HELP poot_prefix_cache_query_tokens_total Prompt tokens looked up against the prefix cache.\n\
             # TYPE poot_prefix_cache_query_tokens_total counter\n\
             poot_prefix_cache_query_tokens_total {}\n\
             # HELP poot_gpu_weight_bytes Model weight bytes resident in GPU memory.\n\
             # TYPE poot_gpu_weight_bytes gauge\n\
             poot_gpu_weight_bytes {}\n\
             # HELP poot_gpu_kv_cache_bytes KV-pool bytes resident in GPU memory.\n\
             # TYPE poot_gpu_kv_cache_bytes gauge\n\
             poot_gpu_kv_cache_bytes {}\n\
             # HELP poot_gpu_resident_bytes Resident GPU memory the engine tracks (weights + KV pool).\n\
             # TYPE poot_gpu_resident_bytes gauge\n\
             poot_gpu_resident_bytes {}\n\
             # HELP poot_gpu_vram_used_bytes Device-wide VRAM in use (bytes), queried from the active backend's runtime GPU API (Vulkan VK_EXT_memory_budget on wgpu, CUDA cuMemGetInfo on PTX, HSA hsa_agent_get_info MEMORY_AVAIL on ROCm); engine-produced, no sysfs/CLI.\n\
             # TYPE poot_gpu_vram_used_bytes gauge\n\
             poot_gpu_vram_used_bytes {}\n\
             # HELP poot_gpu_vram_total_bytes Total device VRAM visible to this process (bytes), from the active backend's runtime GPU API (Vulkan VK_EXT_memory_budget heap_budget on wgpu, CUDA cuMemGetInfo total on PTX, HSA memory-pool SIZE on ROCm); engine-produced, no sysfs/CLI.\n\
             # TYPE poot_gpu_vram_total_bytes gauge\n\
             poot_gpu_vram_total_bytes {}\n\
             # HELP poot_gpu_active_ratio Engine-produced GPU-active fraction (GPU-step time / total decode-step wall time), 0..1.\n\
             # TYPE poot_gpu_active_ratio gauge\n\
             poot_gpu_active_ratio {}\n\
             # HELP poot_decode_dispatch_count Per-token GPU dispatch count of the optimized decode graph (engine-produced; the launches capture/replay collapses to one host call).\n\
             # TYPE poot_decode_dispatch_count gauge\n\
             poot_decode_dispatch_count {}\n\
             # HELP poot_decode_activation_peak_bytes Peak transient activation bytes of the optimized decode graph (engine-produced liveness analysis).\n\
             # TYPE poot_decode_activation_peak_bytes gauge\n\
             poot_decode_activation_peak_bytes {}\n\
             # HELP poot_prefill_dispatch_count GPU dispatch count of the optimized prefill (TTFT) graph at the KV capacity (engine-produced).\n\
             # TYPE poot_prefill_dispatch_count gauge\n\
             poot_prefill_dispatch_count {}\n\
             # HELP poot_prefill_activation_peak_bytes Peak transient activation bytes of the optimized prefill graph (engine-produced; flash keeps attention O(L*head_dim)).\n\
             # TYPE poot_prefill_activation_peak_bytes gauge\n\
             poot_prefill_activation_peak_bytes {}\n\
             # HELP poot_speculative_forwarded_tokens_total Draft tokens forwarded to the batched speculative-decode verify step (proposed, before accept/reject), by sampler mode (spec 268 Phase D).\n\
             # TYPE poot_speculative_forwarded_tokens_total counter\n\
             poot_speculative_forwarded_tokens_total{{mode=\"greedy\"}} {}\n\
             poot_speculative_forwarded_tokens_total{{mode=\"sampled\"}} {}\n\
             # HELP poot_speculative_committed_tokens_total Tokens committed to a slot via the batched speculative-decode path (accepted draft prefix + bonus token), by sampler mode (spec 268 Phase D).\n\
             # TYPE poot_speculative_committed_tokens_total counter\n\
             poot_speculative_committed_tokens_total{{mode=\"greedy\"}} {}\n\
             poot_speculative_committed_tokens_total{{mode=\"sampled\"}} {}\n\
             {}{}{}{}",
            self.started.elapsed().as_secs(),
            self.requests.load(Ordering::Relaxed),
            self.completion_tokens.load(Ordering::Relaxed),
            self.prompt_tokens.load(Ordering::Relaxed),
            self.requests_completed.load(Ordering::Relaxed),
            self.requests_cancelled.load(Ordering::Relaxed),
            self.requests_errored.load(Ordering::Relaxed),
            self.requests_preempted.load(Ordering::Relaxed),
            self.inflight.load(Ordering::Relaxed),
            self.queue_depth(),
            self.batch_size.load(Ordering::Relaxed),
            self.slot_capacity,
            if self.on_gpu { "gpu" } else { "cpu" },
            cpu,
            mb(sys.used_memory()),
            mb(sys.total_memory()),
            (self.kv_cache_usage() * 1000.0).round() / 1000.0,
            self.cache.kv_used_blocks.load(Ordering::Relaxed),
            self.cache.kv_total_blocks.load(Ordering::Relaxed),
            (self.prefix_cache_hit_rate() * 1000.0).round() / 1000.0,
            self.cache.prefix_hit_tokens.load(Ordering::Relaxed),
            self.cache.prefix_query_tokens.load(Ordering::Relaxed),
            self.cache.gpu_weight_bytes.load(Ordering::Relaxed),
            self.cache.gpu_kv_bytes.load(Ordering::Relaxed),
            self.gpu_resident_bytes(),
            self.cache.gpu_vram_used_bytes.load(Ordering::Relaxed),
            self.cache.gpu_vram_total_bytes.load(Ordering::Relaxed),
            (self.gpu_active_ratio() * 1000.0).round() / 1000.0,
            self.cache.decode_dispatch_count.load(Ordering::Relaxed),
            self.cache
                .decode_activation_peak_bytes
                .load(Ordering::Relaxed),
            self.cache.prefill_dispatch_count.load(Ordering::Relaxed),
            self.cache
                .prefill_activation_peak_bytes
                .load(Ordering::Relaxed),
            self.speculative.forwarded_greedy.load(Ordering::Relaxed),
            self.speculative.forwarded_sampled.load(Ordering::Relaxed),
            self.speculative.committed_greedy.load(Ordering::Relaxed),
            self.speculative.committed_sampled.load(Ordering::Relaxed),
            self.ttft
                .prometheus("poot_ttft_ms", "Time to first token (streaming requests)."),
            self.itl
                .prometheus("poot_itl_ms", "Inter-token latency (streaming requests)."),
            self.e2e.prometheus(
                "poot_request_e2e_ms",
                "End-to-end generation request latency."
            ),
            self.queue_wait.prometheus(
                "poot_queue_wait_ms",
                "Time a request waited enqueued before a decode slot freed (GPU batched path)."
            ),
        )
    }
}
