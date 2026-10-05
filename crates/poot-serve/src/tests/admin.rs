use super::*;

#[test]
fn prometheus_exposition_is_well_formed() {
    // model-free: the Prometheus text format carries the counters/gauges with HELP+TYPE lines and the
    // labelled decode-path gauge, reflecting the atomic counter values.
    let m = Metrics {
        started: Instant::now(),
        requests: AtomicU64::new(5),
        completion_tokens: AtomicU64::new(42),
        prompt_tokens: AtomicU64::new(17),
        requests_completed: AtomicU64::new(3),
        requests_cancelled: AtomicU64::new(1),
        requests_errored: AtomicU64::new(2),
        requests_preempted: AtomicU64::new(0),
        inflight: AtomicU64::new(2),
        ttft: Histogram::new(),
        itl: Histogram::new(),
        e2e: Histogram::new(),
        queue_wait: Histogram::new(),
        batch_size: AtomicU64::new(3),
        slot_capacity: 4,
        on_gpu: true,
        profile: false,
        model: "test-model".into(),
        sys: Mutex::new(sysinfo::System::new()),
        // card 051: 4 of 10 blocks used (0.4); 24 of 64 prompt tokens served from cache (0.375).
        cache: CacheMetrics {
            kv_used_blocks: AtomicU64::new(4),
            kv_total_blocks: AtomicU64::new(10),
            prefix_hit_tokens: AtomicU64::new(24),
            prefix_query_tokens: AtomicU64::new(64),
            gpu_weight_bytes: AtomicU64::new(2_000),
            gpu_kv_bytes: AtomicU64::new(512),
            // 800ns of GPU step time out of 1000ns total -> a 0.8 GPU-active ratio.
            decode_gpu_ns: AtomicU64::new(800),
            decode_step_ns: AtomicU64::new(1_000),
            gpu_vram_used_bytes: AtomicU64::new(123_456),
            gpu_vram_total_bytes: AtomicU64::new(987_654),
            decode_dispatch_count: AtomicU64::new(105),
            decode_activation_peak_bytes: AtomicU64::new(4096),
            prefill_dispatch_count: AtomicU64::new(140),
            prefill_activation_peak_bytes: AtomicU64::new(65536),
        },
        // spec 268 Phase D: 10 greedy drafts forwarded, 7 committed (accept prefix + bonus); 6 sampled
        // drafts forwarded, 2 committed.
        speculative: SpeculativeMetrics {
            forwarded_greedy: AtomicU64::new(10),
            committed_greedy: AtomicU64::new(7),
            forwarded_sampled: AtomicU64::new(6),
            committed_sampled: AtomicU64::new(2),
        },
    };
    m.record_request(40.0, 17, Some(8.0));
    let out = m.prometheus();
    assert!(out.contains("# TYPE poot_requests_total counter"));
    assert!(out.contains("poot_requests_total 5"));
    assert!(out.contains("poot_completion_tokens_total 42"));
    assert!(out.contains("poot_prompt_tokens_total 34")); // 17 initial + 17 from record_request
    // card 051: request-outcome counters (completed/cancelled/errored), each its own counter series.
    assert!(out.contains("# HELP poot_requests_completed_total"));
    assert!(out.contains("# TYPE poot_requests_completed_total counter"));
    assert!(out.contains("poot_requests_completed_total 3"));
    assert!(out.contains("# HELP poot_requests_cancelled_total"));
    assert!(out.contains("# TYPE poot_requests_cancelled_total counter"));
    assert!(out.contains("poot_requests_cancelled_total 1"));
    assert!(out.contains("# HELP poot_requests_errored_total"));
    assert!(out.contains("# TYPE poot_requests_errored_total counter"));
    assert!(out.contains("poot_requests_errored_total 2"));
    assert!(out.contains("poot_inflight_requests 2"));
    // inflight 2, batch_size 3 -> saturating_sub = 0 waiting (more decoding than in-system is a transient
    // gauge skew, clamped to 0).
    assert!(out.contains("poot_queue_depth 0"));
    assert!(out.contains("poot_batch_size 3"));
    assert!(out.contains("poot_slot_capacity 4"));
    assert!(out.contains("# TYPE poot_itl_ms histogram"));
    assert!(out.contains("poot_decode_path{path=\"gpu\"} 1"));
    assert!(out.contains("# HELP poot_uptime_seconds"));
    // the e2e histogram series: 40ms lands in the le=50 bucket, count=1, +Inf=count.
    assert!(out.contains("# TYPE poot_request_e2e_ms histogram"));
    assert!(out.contains("poot_request_e2e_ms_count 1"));
    assert!(out.contains("poot_request_e2e_ms_bucket{le=\"+Inf\"} 1"));
    assert!(out.contains("poot_ttft_ms_count 1"));
    // card 051: paged-KV utilization + prefix-cache hit-rate series.
    assert!(out.contains("# TYPE poot_kv_cache_usage_ratio gauge"));
    assert!(out.contains("poot_kv_cache_usage_ratio 0.4"));
    assert!(out.contains("poot_kv_cache_blocks_used 4"));
    assert!(out.contains("poot_kv_cache_blocks_total 10"));
    assert!(out.contains("poot_prefix_cache_hit_rate 0.375"));
    assert!(out.contains("poot_prefix_cache_hit_tokens_total 24"));
    assert!(out.contains("poot_prefix_cache_query_tokens_total 64"));
    // card 030/051: resident GPU memory (weights + KV pool).
    assert!(out.contains("poot_gpu_weight_bytes 2000"));
    assert!(out.contains("poot_gpu_kv_cache_bytes 512"));
    assert!(out.contains("poot_gpu_resident_bytes 2512")); // weights + KV
    assert!(out.contains("poot_gpu_vram_used_bytes 123456")); // device VRAM via VK_EXT_memory_budget
    assert!(out.contains("poot_gpu_vram_total_bytes 987654")); // total VRAM budget (VK_EXT_memory_budget)
    // card 030: the engine-produced GPU-active ratio (800ns GPU / 1000ns step = 0.8). poot's own
    // measurement - no vendor API, no sysfs - so the gauge name is identical on any backend.
    assert!(out.contains("# TYPE poot_gpu_active_ratio gauge"));
    assert!(out.contains("poot_gpu_active_ratio 0.8"));
    // spec 268 Phase D: batched-engine speculative-decode forwarded/committed counters, one series per
    // sampler mode (greedy vs sampled).
    assert!(out.contains("# HELP poot_speculative_forwarded_tokens_total"));
    assert!(out.contains("# TYPE poot_speculative_forwarded_tokens_total counter"));
    assert!(out.contains("poot_speculative_forwarded_tokens_total{mode=\"greedy\"} 10"));
    assert!(out.contains("poot_speculative_forwarded_tokens_total{mode=\"sampled\"} 6"));
    assert!(out.contains("# HELP poot_speculative_committed_tokens_total"));
    assert!(out.contains("# TYPE poot_speculative_committed_tokens_total counter"));
    assert!(out.contains("poot_speculative_committed_tokens_total{mode=\"greedy\"} 7"));
    assert!(out.contains("poot_speculative_committed_tokens_total{mode=\"sampled\"} 2"));

    // queue_depth = inflight - batch_size (vLLM num_requests_waiting), saturating.
    m.inflight.store(5, Ordering::Relaxed);
    m.batch_size.store(2, Ordering::Relaxed);
    assert_eq!(m.queue_depth(), 3, "5 in-system, 2 decoding -> 3 waiting");
    m.batch_size.store(7, Ordering::Relaxed);
    assert_eq!(
        m.queue_depth(),
        0,
        "more decoding than in-system -> saturating 0"
    );
}

/// The typed failure payload is the exact-once ownership boundary shared by every production sender.
/// Delivery does not own accounting: a dropped receiver still records once, while a delivered failure is
/// already marked by its type and the response boundary must not add a second increment. The table also
/// drives the shared rejection, backend active-slot, and queued/active/preempted shutdown senders.
#[test]
fn generation_terminal_failure_accounting_boundary_is_exact_once() {
    let cases = [
        GenerationFailureAccountingCase::DeliveredDirect,
        GenerationFailureAccountingCase::DroppedDirect,
        GenerationFailureAccountingCase::DeliveredRejection,
        GenerationFailureAccountingCase::DroppedRejection,
        GenerationFailureAccountingCase::ActiveFailure,
        GenerationFailureAccountingCase::QueuedShutdown,
        GenerationFailureAccountingCase::ActiveShutdown,
        GenerationFailureAccountingCase::PreemptedShutdown,
    ];

    for case in cases {
        let metrics = test_metrics("terminal-accounting");
        let delivered = !matches!(
            case,
            GenerationFailureAccountingCase::DroppedDirect
                | GenerationFailureAccountingCase::DroppedRejection
        );
        let (reply, reply_rx) = mpsc::channel();
        let mut reply_rx = Some(reply_rx);
        if !delivered {
            drop(reply_rx.take());
        }

        let job = || Job {
            prompt: "failure accounting".into(),
            max_new: 1,
            sampler: Sampler::greedy(),
            stop: vec![],
            stream: false,
            reply,
            submitted: Instant::now(),
            lora_adapter: LoraAdapterLease::none(),
            mrope_positions: None,
        };
        match case {
            GenerationFailureAccountingCase::DeliveredDirect
            | GenerationFailureAccountingCase::DroppedDirect => {
                let _ = job()
                    .reply
                    .send(GenEvent::failed(&metrics, "direct failure"));
            }
            GenerationFailureAccountingCase::DeliveredRejection
            | GenerationFailureAccountingCase::DroppedRejection => {
                reject_job(job(), &metrics, "rejected");
            }
            GenerationFailureAccountingCase::ActiveFailure
            | GenerationFailureAccountingCase::ActiveShutdown
            | GenerationFailureAccountingCase::PreemptedShutdown => {
                let slot = admit_job_to_slot(job(), vec![1], 0, 1, true, None);
                let mut slots = vec![Some(slot)];
                match case {
                    GenerationFailureAccountingCase::ActiveFailure => {
                        fail_active_slots(&mut slots, &metrics, "active failure");
                    }
                    GenerationFailureAccountingCase::ActiveShutdown => {
                        let (job_tx, job_rx) = mpsc::channel();
                        drop(job_tx);
                        shutdown_batch_requests(
                            &job_rx,
                            &mut slots,
                            None,
                            &metrics,
                            "active shutdown",
                        );
                    }
                    GenerationFailureAccountingCase::PreemptedShutdown => {
                        let mut paged = PagedKvCache::new(1, 1);
                        let mut preempted =
                            VecDeque::from([preempt_slot(&mut slots, &mut paged, 0)]);
                        let (job_tx, job_rx) = mpsc::channel();
                        drop(job_tx);
                        shutdown_batch_requests(
                            &job_rx,
                            &mut slots,
                            Some(&mut preempted),
                            &metrics,
                            "preempted shutdown",
                        );
                    }
                    _ => unreachable!("active failure table branch"),
                }
            }
            GenerationFailureAccountingCase::QueuedShutdown => {
                let (job_tx, job_rx) = mpsc::channel();
                job_tx.send(job()).expect("queue shutdown job");
                drop(job_tx);
                let mut slots = vec![None];
                shutdown_batch_requests(&job_rx, &mut slots, None, &metrics, "queued shutdown");
            }
        }

        assert_eq!(
            metrics.requests_errored.load(Ordering::Relaxed),
            1,
            "{case:?}: producer construction owns one increment even if delivery fails"
        );
        if delivered {
            let failure = match reply_rx
                .take()
                .expect("delivered receiver")
                .recv()
                .expect("delivered failure event")
            {
                GenEvent::Failed(failure) => failure,
                _ => panic!("{case:?}: expected Failed"),
            };
            account_generation_error(&metrics, &GenerationError::EngineFailed(failure));
            assert_eq!(
                metrics.requests_errored.load(Ordering::Relaxed),
                1,
                "{case:?}: the response boundary must not double-count producer failure"
            );
        }
        assert_eq!(metrics.requests_completed.load(Ordering::Relaxed), 0);
    }
}

/// Card 326's loopback contract invokes the two real streaming handlers. The token `a` is deliberately
/// held as a strict prefix of stop `ab`: failure and EOF must return before `StreamStops::finish`, while
/// explicit `Done` must retain the established successful tail/finish/usage/`[DONE]` behavior.
#[test]
fn generation_terminal_real_handlers_stop_before_success_finalization() {
    let model_path = write_tiny_granitemoe_gguf();
    let runner = Arc::new(Runner::load_gguf(&model_path).expect("load terminal handler runner"));
    let _ = std::fs::remove_file(&model_path);

    for endpoint in [
        GenerationStreamEndpoint::Completion,
        GenerationStreamEndpoint::Chat,
    ] {
        for terminal in [
            GenerationStreamTerminal::FailedBeforeToken,
            GenerationStreamTerminal::FailedAfterHeldToken,
            GenerationStreamTerminal::EofBeforeToken,
            GenerationStreamTerminal::EofAfterHeldToken,
        ] {
            let (response, metrics) =
                run_generation_stream_handler(endpoint, terminal, Arc::clone(&runner));
            assert!(
                response.starts_with("HTTP/1.1 200 OK\r\n"),
                "{endpoint:?} {terminal:?}: {response}"
            );
            assert!(
                response.contains("\"type\":\"server_error\""),
                "{endpoint:?} {terminal:?}: {response}"
            );
            let expected_code = if matches!(
                terminal,
                GenerationStreamTerminal::FailedBeforeToken
                    | GenerationStreamTerminal::FailedAfterHeldToken
            ) {
                500
            } else {
                503
            };
            assert!(
                response.contains(&format!("\"code\":{expected_code}")),
                "{endpoint:?} {terminal:?}: {response}"
            );
            assert!(
                !response.contains("\"content\":\"a\"") && !response.contains("\"text\":\"a\""),
                "{endpoint:?} {terminal:?}: a held stop-prefix tail leaked: {response}"
            );
            assert!(
                !response.contains("\"finish_reason\":\""),
                "{endpoint:?} {terminal:?}: {response}"
            );
            assert!(!response.contains("\"usage\""), "{endpoint:?} {terminal:?}");
            assert!(
                !response.contains("data: [DONE]"),
                "{endpoint:?} {terminal:?}"
            );
            assert_eq!(metrics.requests_errored.load(Ordering::Relaxed), 1);
            assert_eq!(metrics.requests_completed.load(Ordering::Relaxed), 0);
        }

        let (success, success_metrics) = run_generation_stream_handler(
            endpoint,
            GenerationStreamTerminal::DoneAfterHeldToken,
            Arc::clone(&runner),
        );
        assert!(
            success.contains("\"content\":\"a\"") || success.contains("\"text\":\"a\""),
            "{endpoint:?}: Done must flush the held tail: {success}"
        );
        assert!(
            success.contains("\"finish_reason\":\"length\""),
            "{endpoint:?}"
        );
        assert!(success.contains("\"usage\""), "{endpoint:?}");
        assert!(success.contains("data: [DONE]"), "{endpoint:?}");
        assert_eq!(
            success_metrics.requests_completed.load(Ordering::Relaxed),
            1
        );
        assert_eq!(success_metrics.requests_errored.load(Ordering::Relaxed), 0);

        let (cancelled, cancelled_metrics) = run_generation_stream_handler(
            endpoint,
            GenerationStreamTerminal::Cancelled,
            Arc::clone(&runner),
        );
        assert!(
            !cancelled.contains("\"error\":"),
            "{endpoint:?}: {cancelled}"
        );
        assert!(!cancelled.contains("\"finish_reason\":\""), "{endpoint:?}");
        assert!(!cancelled.contains("\"usage\""), "{endpoint:?}");
        assert!(!cancelled.contains("data: [DONE]"), "{endpoint:?}");
        assert_eq!(
            cancelled_metrics.requests_cancelled.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            cancelled_metrics.requests_completed.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            cancelled_metrics.requests_errored.load(Ordering::Relaxed),
            0
        );
    }
}

#[test]
fn rate_limiter_windows_per_key_and_resets() {
    // card 058 admission control: rpm=2 allows 2 reqs/key/window, rejects the 3rd, isolates keys,
    // and resets after 60s. rpm=0 is unlimited.
    let rl = RateLimiter {
        rpm: 2,
        windows: Mutex::new(HashMap::new()),
    };
    let t0 = Instant::now();
    assert!(rl.check_at("a", t0).is_ok());
    assert!(rl.check_at("a", t0).is_ok());
    // 3rd within the window is rejected with a positive Retry-After.
    match rl.check_at("a", t0) {
        Err(retry) => assert!((1..=60).contains(&retry)),
        Ok(()) => panic!("expected rate-limit rejection"),
    }
    // a different key has its own budget.
    assert!(rl.check_at("b", t0).is_ok());
    // after the window rolls over, key "a" is allowed again.
    assert!(rl.check_at("a", t0 + Duration::from_secs(61)).is_ok());

    // rpm=0 disables the limiter entirely.
    let off = RateLimiter {
        rpm: 0,
        windows: Mutex::new(HashMap::new()),
    };
    for _ in 0..1000 {
        assert!(off.check_at("a", t0).is_ok());
    }
}

#[test]
fn rate_limiter_evicts_stale_keys_but_not_a_live_one() {
    // Regression: `windows` grew forever (one entry per key ever seen), an unbounded-memory DoS if keys were attacker-controlled. `check_at` sweeps entries older than STALE_WINDOW_AGE on every call.
    let rl = RateLimiter {
        rpm: 2,
        windows: Mutex::new(HashMap::new()),
    };
    let t0 = Instant::now();
    // seed 500 distinct one-off keys at t0, then never touch them again.
    for i in 0..500 {
        assert!(rl.check_at(&format!("stale-{i}"), t0).is_ok());
    }
    assert_eq!(rl.windows.lock().unwrap().len(), 500);
    // a key that stays active gets touched again well after the seeded keys.
    let t_mid = t0 + Duration::from_secs(120);
    assert!(rl.check_at("live", t_mid).is_ok());
    // past STALE_WINDOW_AGE (300s) for the seeded keys (last touched at t0, now 350s old) but still
    // within it for "live" (last touched at t_mid=120s, now only 230s old): the sweep must drop the 500
    // stale entries and keep "live".
    let t_far = t0 + Duration::from_secs(350);
    assert!(rl.check_at("live", t_far).is_ok());
    let windows = rl.windows.lock().unwrap();
    assert_eq!(
        windows.len(),
        1,
        "500 stale one-off keys must be evicted, leaving only the still-active \"live\" key"
    );
    assert!(windows.contains_key("live"));
}

/// Card 222: `batch_engine_loop`/`batch_engine_loop_rocm` discarded the terminal `Done` send result and always bumped `requests_completed`, counting a disconnected client's request as completed rather than cancelled.
/// `record_terminal` is the shared card-051 decision the four call sites route through: a delivered `Done` (send_ok=true) is a completion, an undelivered one (send_ok=false, receiver dropped) is a cancellation.
/// Reverting a call site to an unconditional `requests_completed.fetch_add` would make the `send_ok=false` case assert `requests_completed == 1` and fail.
#[test]
fn record_terminal_delivered_is_completed_undelivered_is_cancelled() {
    let m = test_metrics("test-model");
    m.record_terminal(true);
    assert_eq!(m.requests_completed.load(Ordering::Relaxed), 1);
    assert_eq!(m.requests_cancelled.load(Ordering::Relaxed), 0);

    let m = test_metrics("test-model");
    m.record_terminal(false);
    assert_eq!(m.requests_completed.load(Ordering::Relaxed), 0);
    assert_eq!(m.requests_cancelled.load(Ordering::Relaxed), 1);
}

/// Card 222 companion: the disconnect mechanics the batched loops rely on. A dropped `mpsc::Receiver` (a gone client) makes `Sender::send` return `Err`, and that `is_ok()` is what the four terminal-`Done` sites pass into `record_terminal`.
#[test]
fn record_terminal_from_dropped_receiver_send() {
    let (tx, rx) = mpsc::channel::<GenEvent>();
    drop(rx);
    let send_ok = tx.send(GenEvent::Done(vec![1, 2, 3], vec![], true)).is_ok();
    assert!(!send_ok);

    let m = test_metrics("test-model");
    m.record_terminal(send_ok);
    assert_eq!(m.requests_completed.load(Ordering::Relaxed), 0);
    assert_eq!(m.requests_cancelled.load(Ordering::Relaxed), 1);
}

/// Card 315's authorization matrix. It uses the real request parser, policy gate, admin router and response writer over loopback TCP, with a fake loader/pool behind the narrow production trait. Bypassing the guard makes denied Load/Unload rows mutate the snapshot and fail.
#[test]
fn lora_admin_trust_boundary_policy_precedes_every_backend_operation() {
    use std::net::{IpAddr, Ipv4Addr};

    const ADMIN_KEY: &str = "card315-admin-secret";
    const SENTINEL_PATH: &str = "/must-not-be-read/unauthorized-adapter";
    let loopback = IpAddr::V4(Ipv4Addr::LOCALHOST);
    let non_loopback = IpAddr::V4(Ipv4Addr::UNSPECIFIED);

    #[derive(Clone, Copy, Debug)]
    struct PolicyCase {
        name: &'static str,
        listener_ip: IpAddr,
        configured_key: Option<&'static str>,
        authorization: Option<&'static str>,
        expected_status: u16,
        allowed: bool,
    }

    let policy_cases = [
        PolicyCase {
            name: "loopback local trust",
            listener_ip: loopback,
            configured_key: None,
            authorization: None,
            expected_status: 200,
            allowed: true,
        },
        PolicyCase {
            name: "non-loopback disabled",
            listener_ip: non_loopback,
            configured_key: None,
            authorization: Some("Bearer ignored-while-disabled"),
            expected_status: 403,
            allowed: false,
        },
        PolicyCase {
            name: "loopback valid bearer",
            listener_ip: loopback,
            configured_key: Some(ADMIN_KEY),
            authorization: Some("Bearer card315-admin-secret"),
            expected_status: 200,
            allowed: true,
        },
        PolicyCase {
            name: "non-loopback valid bearer",
            listener_ip: non_loopback,
            configured_key: Some(ADMIN_KEY),
            authorization: Some("bearer card315-admin-secret"),
            expected_status: 200,
            allowed: true,
        },
    ];
    let operations = [
        AdminOperation::List,
        AdminOperation::Load,
        AdminOperation::Unload,
    ];

    for policy_case in policy_cases {
        for operation in operations {
            let policy = LoraAdminPolicy::from_configured_key(
                policy_case.listener_ip,
                policy_case.configured_key,
            )
            .unwrap_or_else(|e| panic!("{}: policy: {e:#}", policy_case.name));
            let resolver = FakeLoraAdminResolver::with_base_adapter();
            let before = resolver.admin.snapshot();
            let (method, path, body) = operation.request(SENTINEL_PATH);
            let authorization = policy_case.authorization.into_iter().collect::<Vec<_>>();
            let response =
                run_fake_lora_admin_socket(&policy, &resolver, method, path, &authorization, &body);
            assert!(
                response.starts_with(&format!("HTTP/1.1 {} ", policy_case.expected_status)),
                "{} {operation:?}: {response}",
                policy_case.name
            );
            assert!(
                !response.contains(ADMIN_KEY),
                "{} {operation:?}: credentials must not appear in responses",
                policy_case.name
            );

            let after = resolver.admin.snapshot();
            if policy_case.allowed {
                assert_eq!(resolver.resolve_calls.load(Ordering::SeqCst), 1);
                match operation {
                    AdminOperation::List => assert_eq!(after.list_calls, 1),
                    AdminOperation::Load => {
                        assert_eq!(after.load_paths, [SENTINEL_PATH]);
                        assert_eq!(after.adapters, ["base", "loaded"]);
                    }
                    AdminOperation::Unload => {
                        assert_eq!(after.lookup_calls, 1);
                        assert_eq!(after.unload_calls, 1);
                        assert!(after.adapters.is_empty());
                    }
                }
            } else {
                assert_eq!(
                    resolver.resolve_calls.load(Ordering::SeqCst),
                    0,
                    "{} {operation:?}: denied administration must not select a backend",
                    policy_case.name
                );
                assert_eq!(
                    after, before,
                    "{} {operation:?}: denied administration must be inert",
                    policy_case.name
                );
                assert!(!response.contains(SENTINEL_PATH));
            }
        }
    }

    let denied_credentials: [(&str, &[&str]); 4] = [
        ("missing", &[]),
        ("wrong", &["Bearer wrong-secret"]),
        ("wrong scheme", &["Basic card315-admin-secret"]),
        ("extra token", &["Bearer card315-admin-secret suffix"]),
    ];
    let mut generic_authentication_response = None;
    for listener_ip in [loopback, non_loopback] {
        for (credential_case, authorization) in denied_credentials {
            for operation in operations {
                let policy =
                    LoraAdminPolicy::from_configured_key(listener_ip, Some(ADMIN_KEY)).unwrap();
                let resolver = FakeLoraAdminResolver::with_base_adapter();
                let before = resolver.admin.snapshot();
                let (method, path, body) = operation.request(SENTINEL_PATH);
                let response = run_fake_lora_admin_socket(
                    &policy,
                    &resolver,
                    method,
                    path,
                    authorization,
                    &body,
                );
                assert!(
                    response.starts_with("HTTP/1.1 401 Unauthorized"),
                    "{listener_ip} {credential_case} {operation:?}: {response}"
                );
                assert_eq!(
                    resolver.admin.snapshot(),
                    before,
                    "{listener_ip} {credential_case} {operation:?}: bad credentials must be inert"
                );
                assert_eq!(
                    resolver.resolve_calls.load(Ordering::SeqCst),
                    0,
                    "{listener_ip} {credential_case} {operation:?}: bad credentials must not select a backend"
                );
                assert!(!response.contains(ADMIN_KEY));
                assert!(!response.contains(SENTINEL_PATH));
                let body = response
                    .split_once("\r\n\r\n")
                    .expect("authentication response has a body")
                    .1;
                if let Some(expected) = &generic_authentication_response {
                    assert_eq!(
                        body, expected,
                        "missing, malformed, and incorrect credentials must be indistinguishable"
                    );
                } else {
                    generic_authentication_response = Some(body.to_string());
                }
            }
        }
    }

    #[derive(Clone, Copy)]
    struct WithheldHeadCase {
        name: &'static str,
        method: &'static str,
        path: &'static str,
        authorization: &'static [&'static str],
        content_length: usize,
    }

    let withheld_head_cases = [
        WithheldHeadCase {
            name: "missing authorization before withheld body",
            method: "POST",
            path: "/v1/lora_adapters",
            authorization: &[],
            content_length: 37,
        },
        WithheldHeadCase {
            name: "authorization before body-size response",
            method: "POST",
            path: "/v1/lora_adapters",
            authorization: &[],
            content_length: MAX_BODY_BYTES + 1,
        },
        WithheldHeadCase {
            name: "duplicate invalid then valid",
            method: "POST",
            path: "/v1/lora_adapters",
            authorization: &["Bearer wrong", "Bearer card315-admin-secret"],
            content_length: 37,
        },
        WithheldHeadCase {
            name: "duplicate valid then invalid",
            method: "POST",
            path: "/v1/lora_adapters",
            authorization: &["Bearer card315-admin-secret", "Bearer wrong"],
            content_length: 37,
        },
        WithheldHeadCase {
            name: "exact path",
            method: "GET",
            path: "/v1/lora_adapters",
            authorization: &["Bearer wrong"],
            content_length: 37,
        },
        WithheldHeadCase {
            name: "child path",
            method: "DELETE",
            path: "/v1/lora_adapters/base",
            authorization: &["Bearer wrong"],
            content_length: 37,
        },
        WithheldHeadCase {
            name: "trailing slash",
            method: "GET",
            path: "/v1/lora_adapters/",
            authorization: &["Bearer wrong"],
            content_length: 37,
        },
        WithheldHeadCase {
            name: "query target",
            method: "GET",
            path: "/v1/lora_adapters?limit=1",
            authorization: &["Bearer wrong"],
            content_length: 37,
        },
        WithheldHeadCase {
            name: "unsupported method",
            method: "PATCH",
            path: "/v1/lora_adapters/base?force=true",
            authorization: &["Bearer wrong"],
            content_length: 37,
        },
    ];
    for case in withheld_head_cases {
        let policy = LoraAdminPolicy::from_configured_key(loopback, Some(ADMIN_KEY)).unwrap();
        let (response, resolver) = run_denied_lora_admin_head_without_body(
            policy,
            case.method,
            case.path,
            case.authorization,
            case.content_length,
        );
        assert!(
            response.starts_with("HTTP/1.1 401 Unauthorized"),
            "{}: {response}",
            case.name
        );
        assert_eq!(
            resolver.resolve_calls.load(Ordering::SeqCst),
            0,
            "{}: denial must precede lazy backend selection",
            case.name
        );
        assert_eq!(
            resolver.admin.snapshot(),
            FakeLoraAdmin::with_base_adapter().snapshot(),
            "{}: denial must leave all adapter operations untouched",
            case.name
        );
    }

    #[derive(Clone, Copy)]
    struct SyntaxCase {
        name: &'static str,
        configured: &'static str,
        presented: &'static str,
        allowed: bool,
    }
    let syntax_cases = [
        SyntaxCase {
            name: "RFC punctuation",
            configured: "AbC-._~+/==",
            presented: "Bearer AbC-._~+/==",
            allowed: true,
        },
        SyntaxCase {
            name: "scheme case insensitive",
            configured: "AbC123",
            presented: "bEaReR AbC123",
            allowed: true,
        },
        SyntaxCase {
            name: "multiple SP separator",
            configured: "AbC123",
            presented: "Bearer   AbC123",
            allowed: true,
        },
        SyntaxCase {
            name: "credential case sensitive",
            configured: "AbC123",
            presented: "Bearer abc123",
            allowed: false,
        },
        SyntaxCase {
            name: "invalid punctuation",
            configured: "AbC123",
            presented: "Bearer AbC!23",
            allowed: false,
        },
        SyntaxCase {
            name: "HTAB separator",
            configured: "AbC123",
            presented: "Bearer\tAbC123",
            allowed: false,
        },
        SyntaxCase {
            name: "leading whitespace",
            configured: "AbC123",
            presented: " Bearer AbC123",
            allowed: false,
        },
        SyntaxCase {
            name: "trailing whitespace",
            configured: "AbC123",
            presented: "Bearer AbC123 ",
            allowed: false,
        },
        SyntaxCase {
            name: "internal whitespace",
            configured: "AbC123",
            presented: "Bearer Ab C123",
            allowed: false,
        },
        SyntaxCase {
            name: "Unicode whitespace",
            configured: "AbC123",
            presented: "Bearer\u{a0}AbC123",
            allowed: false,
        },
    ];
    for case in syntax_cases {
        let policy = LoraAdminPolicy::from_configured_key(loopback, Some(case.configured)).unwrap();
        let resolver = FakeLoraAdminResolver::with_base_adapter();
        let response = run_fake_lora_admin_socket(
            &policy,
            &resolver,
            "GET",
            "/v1/lora_adapters",
            &[case.presented],
            "",
        );
        let expected_status = if case.allowed { 200 } else { 401 };
        assert!(
            response.starts_with(&format!("HTTP/1.1 {expected_status} ")),
            "{}: {response}",
            case.name
        );
        assert_eq!(
            resolver.resolve_calls.load(Ordering::SeqCst),
            usize::from(case.allowed),
            "{}: only valid syntax may select a backend",
            case.name
        );
    }

    let supported_shape_cases = [
        ("exact list", "GET", "/v1/lora_adapters", 200),
        ("child wrong method", "GET", "/v1/lora_adapters/base", 404),
        ("trailing slash", "GET", "/v1/lora_adapters/", 404),
        ("query unsupported", "GET", "/v1/lora_adapters?x=1", 404),
        ("unsupported method", "PATCH", "/v1/lora_adapters", 405),
    ];
    for (name, method, path, expected_status) in supported_shape_cases {
        let policy = LoraAdminPolicy::from_configured_key(loopback, Some(ADMIN_KEY)).unwrap();
        let resolver = FakeLoraAdminResolver::with_base_adapter();
        let response = run_fake_lora_admin_socket(
            &policy,
            &resolver,
            method,
            path,
            &["Bearer card315-admin-secret"],
            "",
        );
        assert!(
            response.starts_with(&format!("HTTP/1.1 {expected_status} ")),
            "{name}: {response}"
        );
    }
}

#[test]
fn lora_admin_trust_boundary_rejects_invalid_configured_credentials() {
    use std::net::{IpAddr, Ipv4Addr};

    for invalid in [
        "",
        " ",
        "two words",
        "line\nbreak",
        "tab\tkey",
        "bad!punctuation",
        "bad:punctuation",
        "padding=inside",
        "snowman-☃",
    ] {
        let error =
            LoraAdminPolicy::from_configured_key(IpAddr::V4(Ipv4Addr::LOCALHOST), Some(invalid))
                .err()
                .unwrap_or_else(|| {
                    panic!("invalid credential {invalid:?} must fail startup policy parsing")
                });
        let message = error.to_string();
        assert_eq!(
            message, "POOT_LORA_ADMIN_KEY must be one non-empty RFC 6750 b64token",
            "startup diagnostics must be generic rather than echoing invalid credential contents"
        );
    }

    for valid in ["a", "AbC123", "AbC-._~+/", "token=", "token==="] {
        LoraAdminPolicy::from_configured_key(IpAddr::V4(Ipv4Addr::LOCALHOST), Some(valid))
            .unwrap_or_else(|e| panic!("valid RFC 6750 b64token {valid:?}: {e:#}"));
    }
}

/// The admin policy is not general API authentication. This invokes the complete production HTTP router
/// and completion handler over a socket, using the synthetic in-repo GGUF fixture and a channel-controlled
/// engine. Applying the policy to all POST routes makes both rows return 401/403 before the engine sees a
/// job and fails the test.
#[test]
fn lora_admin_trust_boundary_keeps_ordinary_inference_usable() {
    use std::net::{IpAddr, Ipv4Addr};

    let model_path = write_tiny_granitemoe_gguf();
    let runner =
        Arc::new(Runner::load_gguf(&model_path).expect("load synthetic inference fixture"));
    let _ = std::fs::remove_file(model_path);
    let backend = Arc::new(Backend::Decoder(Arc::clone(&runner)));
    let policies = [
        LoraAdminPolicy::from_configured_key(IpAddr::V4(Ipv4Addr::UNSPECIFIED), None)
            .expect("non-loopback disabled policy"),
        LoraAdminPolicy::from_configured_key(
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            Some("card315-admin-secret"),
        )
        .expect("non-loopback bearer policy"),
    ];

    for policy in policies {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind inference control socket");
        let addr = listener.local_addr().expect("inference socket address");
        let mut client = TcpStream::connect(addr).expect("connect inference control client");
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("bound inference response read");
        let (server, _) = listener.accept().expect("accept inference control client");
        let (job_tx, job_rx) = mpsc::channel();
        let metrics = Arc::new(test_metrics("card315-inference"));
        let limiter = Arc::new(RateLimiter {
            rpm: 0,
            windows: Mutex::new(HashMap::new()),
        });
        let backend_for_handler = Arc::clone(&backend);
        let metrics_for_handler = Arc::clone(&metrics);
        let limiter_for_handler = Arc::clone(&limiter);
        let handler = std::thread::spawn(move || {
            handle(
                server,
                Some(&job_tx),
                &backend_for_handler,
                &metrics_for_handler,
                &limiter_for_handler,
                &policy,
            )
        });

        let body = r#"{"prompt":"a","max_tokens":1}"#;
        write!(
            client,
            "POST /v1/completions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .expect("write inference request without admin credentials");
        client.flush().expect("flush inference request");

        let job = job_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("ordinary inference must reach the engine");
        job.reply
            .send(GenEvent::Done(vec![1], vec![], true))
            .expect("deliver inference completion");
        handler
            .join()
            .expect("join inference handler")
            .expect("handle ordinary inference");

        let mut response = String::new();
        client
            .read_to_string(&mut response)
            .expect("read inference response");
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        assert!(response.contains("text_completion"), "{response}");
        assert!(!response.contains("card315-admin-secret"));
    }
}
