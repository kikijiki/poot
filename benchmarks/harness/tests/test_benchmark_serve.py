"""The serve-throughput scenario (POOT-1334): the load client, the result contract, the cell identity that gains the
concurrency, the manifest cells, one cell through `run_cell`, and the runner's server commands.

The client is driven against a stdlib server that speaks the server-sent-event stream of an OpenAI-style
`/v1/completions`, so every row can fail: a server that fails a request, stops early, or never reports usage.
"""

import http.server
import importlib.util
import json
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

HARNESS_DIR = Path(__file__).resolve().parents[1]
BENCH_ROOT = HARNESS_DIR.parent
sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(BENCH_ROOT / "runners" / "serve"))

import bench  # noqa: E402
import client  # noqa: E402
import observer  # noqa: E402
import performance_receipt as receipts  # noqa: E402
import result_contract as contract  # noqa: E402
from runner_fixtures import POOT_IDENTITY, run_fixture_runner  # noqa: E402
from schema_validator import schema_errors  # noqa: E402
from test_benchmark_compare import DEVICE, DRIVER, HOST_CPU, verdicts  # noqa: E402
from test_perf_page import ROW as PAGE_ROW, generate_classes, write_set  # noqa: E402

spec = importlib.util.spec_from_file_location("serve_run", BENCH_ROOT / "runners" / "serve" / "run.py")
serve_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve_run)

PROMPT = 128
OUTPUT = 128
SCENARIO = {"id": "serve-throughput", "kind": "serve", "concurrency": 4, "prompt_tokens": PROMPT, "output_tokens": OUTPUT,
            "warmup_requests": 8, "requests": 32}


class FakeServer:
    """An OpenAI-style streaming `/v1/completions` on a local port. `tokens` is what it generates when the request does
    not ignore EOS; a request that does ignores it. It counts the requests in flight, keeps every prompt it saw and can
    be told to fail, to stop early or to leave out the usage chunk."""

    def __init__(self, *, bos=1, token_delay=0.002, eos_after=None, fail=None, usage=True, error_event=False):
        self.bos, self.token_delay, self.eos_after, self.fail, self.usage, self.error_event = bos, token_delay, eos_after, fail, usage, error_event
        self.in_flight = self.max_in_flight = 0
        self.prompts, self.bodies = [], []
        self.lock = threading.Lock()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt_tokens = len(body["prompt"].split()) + outer.bos
                with outer.lock:
                    outer.bodies.append(body)
                    outer.prompts.append(body["prompt"])
                    index = len(outer.bodies) - 1
                    if body.get("stream"):
                        outer.in_flight += 1
                        outer.max_in_flight = max(outer.max_in_flight, outer.in_flight)
                try:
                    if outer.fail and outer.fail(index, body):
                        self.send_response(500)
                        self.send_header("Content-Length", "4")
                        self.end_headers()
                        self.wfile.write(b"boom")
                        return
                    if not body.get("stream"):
                        reply = json.dumps({"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1}}).encode()
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(reply)))
                        self.end_headers()
                        self.wfile.write(reply)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    count = body["max_tokens"] if body.get("ignore_eos") or outer.eos_after is None else min(outer.eos_after, body["max_tokens"])
                    for n in range(count):
                        time.sleep(outer.token_delay)
                        chunk = {"choices": [{"index": 0, "text": " w"}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    if outer.error_event:
                        self.wfile.write(b'data: {"error": {"message": "engine fault"}}\n\n')
                    if outer.usage:
                        usage = {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": count}}
                        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                finally:
                    if body.get("stream"):
                        with outer.lock:
                            outer.in_flight -= 1

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def level(server, **overrides):
    args = dict(concurrency=4, prompt_tokens=16, output_tokens=8, warmup_requests=4, requests=12, extra={}, timeout_s=20)
    args.update(overrides)
    return client.run_level("127.0.0.1", server.port, "m", **args)


class LoadClientTests(unittest.TestCase):
    def test_a_request_is_timed_on_its_stream_and_counted_by_the_servers_usage(self):
        with FakeServer() as server:
            record, _ = client.stream_request("127.0.0.1", server.port, client.completion_body("m", "a b c", 6, True, {}), 20)

        self.assertTrue(record["ok"], record)
        self.assertEqual((record["tokens"], record["prompt_tokens"]), (6, 4))
        self.assertGreater(record["ttft_ms"], 0)
        self.assertGreater(record["e2e_ms"], record["ttft_ms"])

    def test_every_request_asks_to_ignore_eos_and_to_stream_a_usage_chunk(self):
        with FakeServer() as server:
            level(server)
        streamed = [b for b in server.bodies if b["stream"]]
        self.assertTrue(streamed)
        for body in streamed:
            self.assertTrue(body["ignore_eos"])
            self.assertEqual(body["stream_options"], {"include_usage": True})
            self.assertEqual(body["temperature"], 0.0)

    def test_a_server_failure_is_a_failed_request_with_its_status_never_a_dropped_one(self):
        with FakeServer(fail=lambda index, body: True) as server:
            record, _ = client.stream_request("127.0.0.1", server.port, client.completion_body("m", "a", 4, True, {}), 20)
        self.assertEqual(record["ok"], False)
        self.assertIn("HTTP 500", record["error"])

    def test_an_error_event_in_the_stream_fails_the_request(self):
        with FakeServer(error_event=True) as server:
            record, _ = client.stream_request("127.0.0.1", server.port, client.completion_body("m", "a", 4, True, {}), 20)
        self.assertEqual(record["ok"], False)
        self.assertIn("engine fault", record["error"])

    def test_a_stream_without_a_usage_chunk_fails_the_request(self):
        with FakeServer(usage=False) as server:
            record, _ = client.stream_request("127.0.0.1", server.port, client.completion_body("m", "a", 4, True, {}), 20)
        self.assertEqual(record["ok"], False)
        self.assertIn("usage", record["error"])

    def test_a_refused_connection_is_a_failed_request(self):
        with FakeServer() as server:
            port = server.port
        record, _ = client.stream_request("127.0.0.1", port, client.completion_body("m", "a", 4, True, {}), 2)
        self.assertEqual(record["ok"], False)

    def test_the_load_keeps_exactly_the_concurrency_in_flight_and_issues_every_request(self):
        for concurrency in (1, 4):
            with self.subTest(concurrency=concurrency), FakeServer(token_delay=0.01) as server:
                result = level(server, concurrency=concurrency, requests=12)
                self.assertEqual(server.max_in_flight, concurrency)
                self.assertEqual(len(result["requests"]), 12)
                self.assertTrue(all(r["ok"] for r in result["requests"]))
                timed = [b for b in server.bodies if b["stream"]][4:]  # the 4 warmup requests came first
                self.assertEqual(len(timed), 12)

    def test_warmup_requests_are_issued_at_the_concurrency_and_are_not_in_the_result(self):
        with FakeServer(token_delay=0.01) as server:
            result = level(server, concurrency=4, warmup_requests=8, requests=4)
        streamed = [b for b in server.bodies if b["stream"]]
        self.assertEqual(len(streamed), 8 + 4)
        self.assertEqual(len(result["requests"]), 4)

    def test_a_failed_warmup_stops_the_cell(self):
        with FakeServer(fail=lambda index, body: body.get("stream")) as server:
            with self.assertRaisesRegex(client.ClientError, "warmup requests failed"):
                level(server)

    def test_the_prompt_is_sized_by_the_servers_own_count_whatever_it_adds(self):
        for bos in (0, 1, 3):
            with self.subTest(bos=bos), FakeServer(bos=bos) as server:
                result = level(server, prompt_tokens=16, requests=2, concurrency=1, warmup_requests=1)
                self.assertEqual({r["prompt_tokens"] for r in result["requests"]}, {16})

    def test_a_prompt_the_server_cannot_hit_exactly_is_an_error_not_a_guess(self):
        with FakeServer() as server:
            with mock.patch.object(client, "probe_prompt_tokens", return_value=7):
                with self.assertRaisesRegex(client.ClientError, "cannot size a prompt of exactly 16 tokens"):
                    level(server, prompt_tokens=16)

    def test_no_two_requests_share_a_prompt_so_no_server_can_answer_from_its_prefix_cache(self):
        with FakeServer() as server:
            level(server, requests=24, warmup_requests=4, concurrency=2)
        prompts = [b["prompt"] for b in server.bodies if b["stream"]]
        self.assertEqual(len(prompts), len(set(prompts)))

    def test_the_server_ignoring_eos_is_what_fixes_the_output_length(self):
        # Without ignore_eos this server stops at its EOS: the request then generates fewer tokens than asked, which
        # the contract refuses (see ContractTests). Here the client's own body is what keeps it to the full length.
        with FakeServer(eos_after=3) as server:
            result = level(server, output_tokens=8, requests=4, warmup_requests=2)
        self.assertEqual({r["tokens"] for r in result["requests"]}, {8})


def serve_request(i=0, *, ttft=20.0, e2e=148.0, tokens=OUTPUT, prompt=PROMPT):
    return {"ok": True, "ttft_ms": ttft + i * 0.1, "e2e_ms": e2e + i * 0.1, "tokens": tokens, "prompt_tokens": prompt}


def serve_payload(framework="poot", *, concurrency=4, requests=None, wall_ms=1000.0, slots=None, **overrides):
    payload = {
        "framework": framework, "precision": "bf16", "mode": "serve", "backend": "ptx" if framework == "poot" else "cuda",
        "concurrency": concurrency, "prompt_tokens": PROMPT, "output_tokens": OUTPUT,
        "server_slots": concurrency if slots is None else slots, "wall_ms": wall_ms,
        "requests": [serve_request(i) for i in range(32)] if requests is None else requests,
    }
    if framework == "poot":
        payload.update({k: v for k, v in POOT_IDENTITY.items() if k != "prefill_chunk"}, family="qwen3")
    else:
        payload["framework_version"] = "fixture-1"
    return {**payload, **overrides}


def validate(payload, framework="poot", **scenario):
    return contract.validate_runner_result(
        payload, framework=framework, precision="bf16", scenario={**SCENARIO, **scenario}, gen_tokens=OUTPUT,
        backend=payload.get("backend") if framework == "poot" else None)


class ContractTests(unittest.TestCase):
    def errors(self, payload, **kwargs):
        with self.assertRaises(contract.ResultValidationError) as raised:
            validate(payload, **kwargs)
        return raised.exception.errors

    def test_a_valid_serve_run_aggregates_the_requests_into_throughput_and_percentiles(self):
        prepared = validate(serve_payload(wall_ms=2000.0))
        summary = prepared["serve"]

        self.assertEqual(summary["output_tok_s"], 32 * OUTPUT / 2.0)
        self.assertEqual((summary["requests"], summary["failed_requests"], summary["concurrency"]), (32, 0, 4))
        self.assertAlmostEqual(summary["ttft_ms"]["p50"], 20.0 + 15.5 * 0.1)
        self.assertAlmostEqual(summary["tpot_ms"]["p50"], (148.0 - 20.0) / (OUTPUT - 1))
        # 32 requests are enough samples for a tail: p90 and p99 are stated.
        self.assertIn("p99", summary["ttft_ms"])
        self.assertIn("p90", summary["tpot_ms"])

    def test_the_timing_claim_of_a_serve_cell_is_the_load_clients_for_every_engine(self):
        for framework in ("poot", "vllm", "llamacpp"):
            self.assertEqual(contract.timing_claim(framework, SCENARIO), contract.CLIENT_STREAM)
        self.assertEqual(contract.timing_claim("poot"), contract.REQUEST_WINDOW)
        self.assertEqual(contract.timing_claim("vllm"), contract.ROW_LEVEL)

    def test_a_failed_request_fails_the_cell_with_the_count_and_the_first_error(self):
        requests = [serve_request(i) for i in range(32)]
        requests[5] = {"ok": False, "error": "HTTP 503: queue full"}
        requests[9] = {"ok": False, "error": "timeout"}
        errors = self.errors(serve_payload(requests=requests))
        self.assertEqual(list(errors), ["2 of 32 requests failed at concurrency 4 (first: HTTP 503: queue full)"])

    def test_a_request_that_stopped_early_is_refused_never_averaged(self):
        requests = [serve_request(i) for i in range(32)]
        requests[3] = serve_request(3, tokens=100)
        errors = self.errors(serve_payload(requests=requests))
        self.assertEqual(list(errors), ["requests[3].generated 100 tokens, expected 128"])

    def test_a_prompt_of_another_length_is_refused(self):
        requests = [serve_request(i) for i in range(32)]
        requests[0] = serve_request(0, prompt=PROMPT + 1)
        self.assertIn("requests[0].prompt was 129 tokens, expected 128", self.errors(serve_payload(requests=requests)))

    def test_a_server_with_fewer_slots_than_the_concurrency_is_not_that_concurrency(self):
        errors = self.errors(serve_payload(concurrency=16, slots=4), concurrency=16)
        self.assertTrue(any("server_slots must be an integer of at least the concurrency 16" in e for e in errors), errors)

    def test_the_run_must_hold_every_timed_request(self):
        errors = self.errors(serve_payload(requests=[serve_request(i) for i in range(31)]))
        self.assertEqual(list(errors), ["requests must hold the 32 timed requests; got 31"])

    def test_a_run_at_another_concurrency_than_the_cells_is_refused(self):
        errors = self.errors(serve_payload(concurrency=1, slots=1), concurrency=4)
        self.assertIn("concurrency mismatch: requested 4, reported 1", errors)

    def test_first_token_and_finish_must_be_ordered(self):
        requests = [serve_request(i) for i in range(32)]
        requests[2] = serve_request(2, ttft=50.0, e2e=50.0)
        self.assertIn("requests[2].e2e_ms must be greater than ttft_ms", self.errors(serve_payload(requests=requests)))

    def test_a_poot_server_states_the_identity_of_what_it_runs_on(self):
        for field in ("adapter", "driver_version", "checkpoint_digest", "compute_dtype", "family", "build_sha"):
            payload = serve_payload()
            del payload[field]
            with self.subTest(field=field):
                self.assertTrue(any(field in e for e in self.errors(payload)), field)

    def test_an_external_server_names_its_version_and_needs_no_build(self):
        validate(serve_payload("vllm"), framework="vllm")
        payload = serve_payload("vllm")
        del payload["framework_version"]
        self.assertTrue(any("framework_version" in e for e in self.errors(payload, framework="vllm")))

    def test_a_serve_result_is_not_a_single_stream_result(self):
        with self.assertRaises(contract.ResultValidationError) as raised:
            validate(serve_payload(mode="single"))
        self.assertTrue(any("mode mismatch" in e for e in raised.exception.errors))


def serve_row(output_tok_s, *, concurrency=4, ttft=20.0, binary="bin-1", model="m", **extra):
    return {
        "framework": "poot", "model": model, "precision": "bf16", "scenario": "serve-throughput", "status": "ok",
        "backend": "ptx", "device": DEVICE, "host_cpu": HOST_CPU, "gpu_driver": DRIVER, "binary_sha256": binary,
        **{k: v for k, v in POOT_IDENTITY.items() if k != "prefill_chunk"}, "build_sha": binary.ljust(40, "0"),
        "concurrency": concurrency, "prompt_tokens": PROMPT, "gen_tokens": OUTPUT, "output_tok_s": output_tok_s,
        "ttft_ms": ttft, "tpot_ms": 5.0, "e2e_ms": 700.0, "peak_vram_bytes": 2**30, **extra,
    }


QUIET = [98.0, 99.0, 100.0, 101.0, 102.0]


def serve_repeats(samples, *, ttft=20.0, **kwargs):
    # Each repeat's TTFT differs a little, as real ones do: identical repeats state no noise to compare against.
    return [serve_row(tok_s, ttft=ttft + jitter, tpot_ms=5.0 + jitter / 10, **kwargs)
            for tok_s, jitter in zip(samples, (-0.4, -0.2, 0.0, 0.2, 0.4))]


class CellIdentityTests(unittest.TestCase):
    def test_two_concurrencies_of_one_engine_are_two_cells_never_one_pooled_band(self):
        rows = serve_repeats(QUIET, concurrency=1) + serve_repeats([x * 4 for x in QUIET], concurrency=4)
        cells = bench.compare_cells(rows, rows, receipts.POLICY)

        self.assertEqual(sorted(cell["key"][7] for cell in cells), [1, 4])
        self.assertEqual({cell["key"][7]: cell["median_a"] for cell in cells}, {1: 100.0, 4: 400.0})
        self.assertEqual({cell["n_a"] for cell in cells}, {5})

    def test_a_loss_at_one_concurrency_is_that_cells_regression_and_not_the_others(self):
        base = serve_repeats(QUIET, concurrency=1) + serve_repeats(QUIET, concurrency=4)
        cand = serve_repeats(QUIET, concurrency=1) + serve_repeats([x * 0.7 for x in QUIET], concurrency=4)

        found = {key[7]: verdict for key, verdict in verdicts(base, cand).items()}

        self.assertEqual(found, {1: receipts.WITHIN_NOISE, 4: receipts.REGRESSION})

    def test_a_serve_cell_is_judged_by_output_tokens_per_second(self):
        # The same rows carry no decode_tok_s: a comparator reading it would call them invalid.
        base, cand = serve_repeats(QUIET), serve_repeats([x * 1.5 for x in QUIET])
        self.assertEqual(list(verdicts(base, cand).values()), [receipts.IMPROVEMENT])

    def test_latency_at_a_fixed_concurrency_is_part_of_the_verdict(self):
        base = serve_repeats(QUIET, ttft=20.0)
        cand = serve_repeats(QUIET, ttft=40.0)
        (cell,) = bench.compare_cells(base, cand, receipts.POLICY)
        self.assertEqual(cell["verdict"], receipts.REGRESSION)
        self.assertIn("latency ttft_ms", cell["note"])

    def test_a_row_with_no_concurrency_keeps_the_key_it_had_with_none_in_the_new_place(self):
        row = {**serve_row(1.0), "scenario": "decode-128"}
        del row["concurrency"]
        self.assertIsNone(bench.cell_key(row)[7])

    def test_a_serve_row_is_identified_without_a_prefill_chunk_or_a_profiled_flag(self):
        fields = receipts.row_identity_fields("poot", serve=True)
        self.assertNotIn("prefill_chunk", fields)
        self.assertNotIn("profiled", fields)
        self.assertIn("checkpoint_digest", fields)
        self.assertIn("prefill_chunk", receipts.row_identity_fields("poot"))


class ManifestCellTests(unittest.TestCase):
    def manifest(self):
        return bench.load_toml(BENCH_ROOT / "manifest.toml")

    def serve_cells(self):
        return [c for c in bench.cells(self.manifest()) if c[1].get("kind") == "serve"]

    def test_each_concurrency_of_the_scenario_is_its_own_cell_with_a_scalar_concurrency(self):
        poot = [c for c in self.serve_cells() if c[0]["id"] == "qwen3-0.6b" and c[2] == "poot"]
        self.assertEqual([c[1]["concurrency"] for c in poot], [1, 4, 16])
        for _, scen, _, _ in poot:
            self.assertNotIn("concurrencies", scen)
            self.assertEqual((scen["prompt_tokens"], scen["output_tokens"]), (128, 128))
            self.assertGreaterEqual(scen["warmup_requests"], 1)
            self.assertGreaterEqual(scen["requests"], 20)

    def test_an_engine_with_no_concurrent_server_is_unsupported_with_the_scenarios_reason(self):
        reasons = {c[2]: c[3] for c in self.serve_cells() if c[0]["id"] == "qwen3-0.6b" and c[1]["concurrency"] == 4}
        for framework in ("candle", "transformers"):
            self.assertFalse(bench.is_supported(reasons[framework]))
            self.assertIn("single-stream", reasons[framework]["reason"])
        self.assertTrue(bench.is_supported(reasons["poot"]))

    def test_every_engine_with_a_server_has_a_serve_command_and_its_binary(self):
        runners = bench.load_toml(BENCH_ROOT / "runners.toml")
        for framework in ("poot", "vllm", "llamacpp"):
            self.assertIn("serve_cmd", runners[framework], framework)
        self.assertIn("serve_bin", runners["poot"])
        self.assertIn("serve_bin", runners["llamacpp"])
        for framework in ("candle", "transformers"):
            self.assertNotIn("serve_cmd", runners[framework])

    def test_the_other_scenarios_are_unchanged_cells(self):
        scenarios = [c[1]["id"] for c in bench.cells(self.manifest()) if c[0]["id"] == "qwen3-0.6b" and c[2] == "poot"]
        self.assertEqual(scenarios.count("serve-throughput"), 3)
        self.assertEqual(scenarios.count("decode-128"), 1)


def executable(directory, name, payload=None):
    path = Path(directory) / name
    path.write_text("#!/bin/sh\n" + (f"printf '%s\\n' '{json.dumps(payload)}'\n" if payload is not None else ""), encoding="ascii")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def run_serve_cell(directory, payload, *, fw="poot", fw_cfg=None, scenario=None):
    server = executable(directory, "poot-serve")
    runner = executable(directory, "serve-runner", payload)
    runners = {fw: {"bin": str(runner), "cmd": ["{bin}"], "serve_bin": str(server), "serve_cmd": ["{bin}"]}}
    with (
        mock.patch.object(observer, "run_observed", side_effect=run_fixture_runner),
        mock.patch.dict(os.environ, {"POOT_BENCH_BACKEND": "ptx"}),
    ):
        return bench.run_cell(
            model={"id": "fixture", "match_precision": "bf16"}, scen=scenario or SCENARIO, fw=fw,
            fw_cfg=fw_cfg or {"precision": "bf16"}, runners=runners, models_dir=".", suite={}, interval_ms=50,
            device="fixture-gpu")


class RunCellTests(unittest.TestCase):
    def test_a_serve_cell_becomes_one_row_of_its_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = run_serve_cell(tmp, serve_payload(wall_ms=2000.0))
            served_digest = bench.binary_sha256(row_server_path(tmp))

        self.assertEqual(row["status"], "ok", row.get("reason"))
        self.assertEqual((row["concurrency"], row["scenario"]), (4, "serve-throughput"))
        self.assertEqual(row["output_tok_s"], 32 * OUTPUT / 2.0)
        self.assertEqual((row["prompt_tokens"], row["gen_tokens"], row["server_slots"]), (PROMPT, OUTPUT, 4))
        self.assertEqual(row["timing_claim"], "client-stream")
        self.assertEqual(row["ttft_ms"], row["serve"]["ttft_ms"]["p50"])
        self.assertEqual(row["family"], "qwen3")
        self.assertEqual(row["build_sha"], POOT_IDENTITY["build_sha"])
        # The binary measured is the server's, not the single-stream runner's.
        self.assertEqual(row["binary_sha256"], served_digest)

    def test_a_serve_row_is_a_row_of_the_schema(self):
        schema = json.loads((BENCH_ROOT / "schema" / "result.schema.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            row = run_serve_cell(tmp, serve_payload())
        self.assertEqual(schema_errors(row, schema), [])
        broken = {**row, "concurrency": 0}
        self.assertTrue(schema_errors(broken, schema))

    def test_a_cell_with_a_failed_request_is_an_error_row_that_says_so(self):
        requests = [serve_request(i) for i in range(32)]
        requests[0] = {"ok": False, "error": "HTTP 500: boom"}
        with tempfile.TemporaryDirectory() as tmp:
            row = run_serve_cell(tmp, serve_payload(requests=requests))
        self.assertEqual(row["status"], "error")
        self.assertIn("1 of 32 requests failed at concurrency 4 (first: HTTP 500: boom)", row["reason"])
        self.assertNotIn("output_tok_s", row)

    def test_a_runner_that_could_not_start_its_server_reports_why(self):
        payload = {"framework": "poot", "mode": "serve", "error": "the server exited with code 1 before it was healthy"}
        with tempfile.TemporaryDirectory() as tmp:
            row = run_serve_cell(tmp, payload)
        self.assertEqual(row["status"], "error")
        self.assertIn("exited with code 1", row["reason"])

    def test_an_engine_the_scenario_declares_without_a_server_runs_nothing(self):
        cells = [c for c in bench.cells(bench.load_toml(BENCH_ROOT / "manifest.toml"))
                 if c[1].get("kind") == "serve" and c[2] == "transformers"][0]
        model, scen, fw, fw_cfg = cells
        row = bench.run_cell(model, scen, fw, fw_cfg, {}, ".", {}, 50)
        self.assertEqual(row["status"], "unsupported")
        self.assertEqual(row["concurrency"], scen["concurrency"])
        self.assertIn("single-stream", row["reason"])


def row_server_path(directory):
    return str(Path(directory) / "poot-serve")


class PageTests(unittest.TestCase):
    def rows(self, concurrency, base):
        served = {k: v for k, v in PAGE_ROW.items() if k != "decode_tok_s"}
        return [{**served, "scenario": "serve-throughput", "concurrency": concurrency, "gen_tokens": 128, "prompt_tokens": 128,
                 "output_tok_s": base + delta, "ttft_ms": 20.0, "tpot_ms": 5.0, "e2e_ms": 700.0}
                for delta in (-1.0, -0.5, 0.0, 0.5, 1.0)]

    def test_each_concurrency_is_a_point_of_its_own_on_the_page_and_none_is_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            # One run directory per repeat; two concurrencies are two cells of the same set.
            for i in range(5):
                run = Path(tmp) / f"serve-r{i + 1}"
                run.mkdir()
                (run / "results.jsonl").write_text("".join(json.dumps(row[i]) + "\n" for row in (self.rows(1, 100.0), self.rows(4, 400.0))))
                (run / "env.json").write_text(json.dumps({"captured_at_utc": "2026-10-21T00:00:00+00:00", "cpu": "Test CPU"}))
            (cls,) = generate_classes(tmp)

        points = sorted(cls["points"], key=lambda p: p["concurrency"])
        self.assertEqual([p["concurrency"] for p in points], [1, 4])
        self.assertEqual([p["metrics"]["output_tok_s"]["median"] for p in points], [100.0, 400.0])
        self.assertTrue(all(p["isl"] == 128 for p in points))
        self.assertTrue(all("decode_tok_s" not in p["metrics"] for p in points))
        self.assertTrue(all(p["latest"] and p["verified"] for p in points))

    def test_a_single_stream_point_carries_no_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_set(Path(tmp), "single", [{**PAGE_ROW, "decode_tok_s": 166.0 + d} for d in (-1.0, -0.5, 0.0, 0.5, 1.0)])
            (cls,) = generate_classes(tmp)
        (point,) = cls["points"]
        self.assertNotIn("concurrency", point)


class Args:
    def __init__(self, framework, **overrides):
        self.framework, self.server_bin, self.model_dir, self.precision, self.backend = framework, "SERVER", "/models/m", "bf16", "wgpu"
        self.concurrency, self.prompt_tokens, self.output_tokens, self.gpu_mem_util = 16, 128, 128, 0.85
        self.__dict__.update(overrides)


class ServerCommandTests(unittest.TestCase):
    def flag(self, cmd, name):
        return cmd[cmd.index(name) + 1]

    def test_every_server_holds_as_many_slots_as_the_concurrency_and_positions_for_each_request(self):
        positions = 128 + 128 + serve_run.MARGIN_TOKENS
        poot = serve_run.server_command(Args("poot"), 9000)
        self.assertEqual((self.flag(poot, "--rows"), self.flag(poot, "--max-seq")), ("16", str(positions)))
        self.assertEqual(self.flag(poot, "--backend"), "wgpu")
        vllm = serve_run.server_command(Args("vllm"), 9000)
        self.assertEqual((self.flag(vllm, "--max-num-seqs"), self.flag(vllm, "--max-model-len")), ("16", str(positions)))
        self.assertIn("--no-enable-prefix-caching", vllm)
        llama = serve_run.server_command(Args("llamacpp"), 9000, "/m/model.gguf")
        self.assertEqual((self.flag(llama, "--parallel"), self.flag(llama, "-c")), ("16", str(16 * positions)))
        self.assertEqual(self.flag(llama, "-m"), "/m/model.gguf")

    def test_a_server_listens_on_loopback_only(self):
        for framework in ("poot", "vllm", "llamacpp"):
            cmd = serve_run.server_command(Args(framework), 9123, "/m/model.gguf")
            self.assertTrue(any("127.0.0.1" in part for part in cmd), framework)
            self.assertFalse(any("0.0.0.0" in part for part in cmd), framework)

    def test_vllm_takes_the_precision_and_a_quantization_where_there_is_one(self):
        self.assertEqual(self.flag(serve_run.server_command(Args("vllm"), 1), "--dtype"), "bfloat16")
        quantized = serve_run.server_command(Args("vllm", precision="gptq-int4"), 1)
        self.assertEqual(self.flag(quantized, "--quantization"), "gptq")

    def test_only_llama_server_gets_its_prompt_cache_switch_in_the_request(self):
        self.assertEqual(serve_run.request_extras("llamacpp"), {"cache_prompt": False})
        self.assertEqual(serve_run.request_extras("poot"), {})
        self.assertEqual(serve_run.request_extras("vllm"), {})

    def test_no_engine_is_a_server_it_does_not_have(self):
        with self.assertRaises(ValueError):
            serve_run.server_command(Args("candle"), 1)

    def test_llama_servers_backend_is_read_from_its_own_log_and_a_cpu_run_is_refused(self):
        self.assertEqual(serve_run.llama_backend("ggml_cuda_init: found 1 CUDA devices:\n"), "cuda")
        self.assertEqual(serve_run.llama_backend("ggml_cuda_init: found 1 ROCm devices:\n"), "rocm")
        self.assertEqual(serve_run.llama_backend("ggml_vulkan: Found 1 Vulkan devices:\n"), "vulkan")
        with self.assertRaisesRegex(client.ClientError, "no accelerator backend"):
            serve_run.llama_backend("load_tensors: CPU model buffer size = 1137.00 MiB\n")

    def test_an_external_server_that_names_no_version_is_refused(self):
        self.assertEqual(serve_run.llama_version({"build_info": "b7000-abc"}), "b7000-abc")
        with self.assertRaises(client.ClientError):
            serve_run.llama_version({})


class IdentityNamesTests(unittest.TestCase):
    """The names a poot server and the poot bench runner state their identity under are one list in Rust
    (`RunIdentity::fields`, `BuildIdentity::fields`); the contract that reads them must name exactly those."""

    def stated(self):
        root = BENCH_ROOT.parent / "crates"
        sources = (root / "poot-llm/src/driver/identity.rs").read_text() + (root / "poot-build-identity/src/lib.rs").read_text()
        import re
        return set(re.findall(r'\("([a-z_]+)",\s', sources))

    def test_the_contract_reads_exactly_the_fields_the_binaries_state(self):
        read = {*contract.BUILD_IDENTITY_FIELDS, *contract.MEASUREMENT_IDENTITY_FIELDS, "family", "precision"}
        stated = self.stated()
        self.assertEqual(read - stated, set(), "the contract reads a field no poot binary states")
        self.assertEqual({n for n in stated if n.islower()} - read, set(), "a binary states a field the contract does not read")


if __name__ == "__main__":
    unittest.main()
