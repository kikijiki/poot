"""The load client of the serve-throughput scenario (POOT-1334): one client, stdlib only, for every engine's server.

It sends streaming `/v1/completions` requests of a fixed prompt length and a fixed output length at a fixed
concurrency and records, per request, what it saw on the stream: the time of the first token, the time of the last,
and the token counts the server reports in its usage chunk. Nothing is timed by the server, so TTFT and TPOT are
measured one way for poot, vLLM and llama.cpp (see NOTES.md for the prior art this follows).

A request that fails is recorded with its error, never dropped: the harness fails the cell on any failed request.
"""

import http.client
import json
import random
import threading
import time

# Words that are one token each, with their leading space, in the byte-level BPE and SentencePiece vocabularies of
# the models the grid runs: a prompt of N of them is N tokens, plus whatever constant the tokenizer adds (a BOS).
# `calibrate_prompt` measures that constant against the server, so the pool never has to know it.
WORDS = (
    "the of and to in is it for on with as at by an be this that from or are was but not have had has were which "
    "their there been would about more other into than them these some could time very when your what can will "
    "up out if so no we our one all new now only over such then its two may first any who also most where after"
).split()
PROBE_SEED = -1
CALIBRATION_ROUNDS = 4


class ClientError(RuntimeError):
    """The load client cannot run the cell at all (a probe failed, the prompt cannot be sized)."""


def prompt_text(words, seed):
    """A prompt of `words` words drawn from `WORDS` by `seed`: different seeds differ in their first words, so no two
    requests share a prefix a server could answer from its cache."""
    rng = random.Random(seed)
    return "".join(" " + rng.choice(WORDS) for _ in range(words)).lstrip()


def completion_body(model, prompt, max_tokens, stream, extra):
    """The request body: greedy, to exactly `max_tokens` (the server ignores EOS), streamed with a usage chunk."""
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True,
            "stream": stream, **extra}
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def sse_events(lines):
    """The JSON payloads of a server-sent-event stream, in order, until `[DONE]`. A line that is not `data:` is skipped."""
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        yield json.loads(payload)


class _Stream:
    """What one request's stream says: when the first and last token chunks arrived and the usage chunk's counts."""

    def __init__(self):
        self.first = self.last = None
        self.usage = None
        self.error = None

    def observe(self, event, now):
        if "error" in event:
            error = event["error"]
            self.error = error.get("message") if isinstance(error, dict) else str(error)
            return
        if event.get("usage"):
            self.usage = event["usage"]
        choices = event.get("choices") or []
        if choices and choices[0].get("text"):
            self.first = self.first if self.first is not None else now
            self.last = now


def stream_request(host, port, body, timeout_s, clock=time.perf_counter):
    """One streaming completion. Returns `(record, started)` where `record` is `{"ok": True, "ttft_ms", "e2e_ms",
    "tokens", "prompt_tokens"}` or `{"ok": False, "error"}`. TTFT is the first chunk that carries text; the end is the
    last chunk that does. `tokens` is the server's own count (`usage.completion_tokens`), so a run that generated
    fewer than asked is visible to the contract."""
    started = clock()
    stream = _Stream()
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        try:
            conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
            response = conn.getresponse()
            if response.status != 200:
                return {"ok": False, "error": f"HTTP {response.status}: {response.read(300).decode('utf-8', 'replace')}"}, started
            for event in sse_events(response):
                stream.observe(event, clock())
        finally:
            conn.close()
    except (OSError, http.client.HTTPException, ValueError) as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}, started
    if stream.error:
        return {"ok": False, "error": f"the server reported an error: {stream.error}"}, started
    if stream.first is None:
        return {"ok": False, "error": "the stream carried no token"}, started
    usage = stream.usage or {}
    if not (isinstance(usage.get("completion_tokens"), int) and isinstance(usage.get("prompt_tokens"), int)):
        return {"ok": False, "error": f"the stream ended without a usage chunk ({usage!r})"}, started
    return {"ok": True, "ttft_ms": (stream.first - started) * 1000.0, "e2e_ms": (stream.last - started) * 1000.0,
            "tokens": usage["completion_tokens"], "prompt_tokens": usage["prompt_tokens"]}, started


def probe_prompt_tokens(host, port, model, prompt, timeout_s):
    """The number of tokens the server counts in `prompt`, from a one-token non-streaming completion."""
    body = completion_body(model, prompt, 1, False, {})
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
        try:
            conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
            response = conn.getresponse()
            text = response.read().decode("utf-8", "replace")
        finally:
            conn.close()
    except (OSError, http.client.HTTPException) as error:
        raise ClientError(f"the prompt-size probe failed: {type(error).__name__}: {error}") from error
    if response.status != 200:
        raise ClientError(f"the prompt-size probe got HTTP {response.status}: {text[:300]}")
    try:
        return json.loads(text)["usage"]["prompt_tokens"]
    except (ValueError, KeyError, TypeError) as error:
        raise ClientError(f"the prompt-size probe's reply has no usage.prompt_tokens: {text[:300]}") from error


def calibrate_prompt(count_tokens, target):
    """The number of words whose prompt the server counts as exactly `target` tokens. `count_tokens(words)` is the
    server's count for a prompt of that many words; the difference (a BOS, a template) is added until it matches."""
    words = target
    for _ in range(CALIBRATION_ROUNDS):
        counted = count_tokens(words)
        if counted == target:
            return words
        words += target - counted
        if words < 1:
            break
    raise ClientError(f"cannot size a prompt of exactly {target} tokens: the server counted {counted} for {words} words")


def run_requests(count, concurrency, issue):
    """Issue `count` requests with `concurrency` in flight (closed loop: a finished request is replaced at once, until
    the last is issued). `issue(index)` runs one request and returns its record. Returns the records in issue order and
    the wall time in ms from the first request's start to the last one's finish."""
    records = [None] * count
    finished = [0.0] * count
    starts = [0.0] * count
    lock = threading.Lock()
    cursor = [0]

    def worker():
        while True:
            with lock:
                index = cursor[0]
                if index >= count:
                    return
                cursor[0] += 1
            record, started = issue(index)
            starts[index] = started
            finished[index] = time.perf_counter()
            records[index] = record

    threads = [threading.Thread(target=worker) for _ in range(min(concurrency, count))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return records, (max(finished) - min(starts)) * 1000.0


def run_level(host, port, model, *, concurrency, prompt_tokens, output_tokens, warmup_requests, requests, extra,
              timeout_s, seed=0, on_warmed=lambda: None):
    """One concurrency of the scenario against a running server: size the prompt, warm up at this concurrency, then time
    `requests` requests (`on_warmed` runs between the two). Returns `{"wall_ms", "requests"}`; the caller adds what
    it knows of the server."""
    words = calibrate_prompt(lambda n: probe_prompt_tokens(host, port, model, prompt_text(n, PROBE_SEED), timeout_s),
                             prompt_tokens)

    def issue(index, base):
        body = completion_body(model, prompt_text(words, base + index), output_tokens, True, extra)
        return stream_request(host, port, body, timeout_s)

    # Warmup requests are distinct from the timed ones (a different seed range) and are not recorded; one that fails
    # stops the cell, since a server that cannot answer them has nothing to measure.
    warm, _ = run_requests(warmup_requests, concurrency, lambda i: issue(i, seed - 10_000_000))
    failed = [r["error"] for r in warm if not r["ok"]]
    if failed:
        raise ClientError(f"{len(failed)} of {warmup_requests} warmup requests failed (first: {failed[0]})")
    on_warmed()
    timed, wall_ms = run_requests(requests, concurrency, lambda i: issue(i, seed))
    return {"wall_ms": wall_ms, "requests": timed}
