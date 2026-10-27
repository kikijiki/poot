"""The fixed generated length of a vLLM request and the per-token timing of its stream (no vllm import, so the
harness tests can drive it without a GPU)."""


class StoppedEarly(RuntimeError):
    """The stream ended before the fixed generated length."""


def fixed_length_kwargs(tokens):
    """SamplingParams arguments that generate exactly `tokens` greedy tokens: no EOS stop, no early finish."""
    return {"min_tokens": tokens, "max_tokens": tokens, "temperature": 0.0, "ignore_eos": True}


class TokenStream:
    """Per-token timing of one streamed request. vLLM delivers a chunk with `total` tokens so far, and a chunk may
    carry several: the first-token boundary is the first chunk, the tokens that arrive with it add zero-length gaps
    (they were delivered at the same instant), and the tokens of a later chunk share that chunk's gap."""

    def __init__(self, start):
        self.start = start
        self.ttft_ms = None
        self.itl_ms = []
        self._last = None
        self._total = 0

    def observe(self, now, total):
        new = total - self._total
        if new <= 0:
            return
        if self.ttft_ms is None:
            self.ttft_ms = (now - self.start) * 1000.0
            self.itl_ms.extend([0.0] * (new - 1))
        else:
            self.itl_ms.extend([(now - self._last) * 1000.0 / new] * new)
        self._last = now
        self._total = total

    def finish(self, now, tokens):
        """(ttft_ms, e2e_ms, itl_ms) of a stream that produced exactly `tokens` tokens; otherwise the framework
        is named with the count it generated and, when short, the token it stopped at."""
        if self._total != tokens:
            ended = f"; the stream ended after token {self._total}" if self._total < tokens else ""
            raise StoppedEarly(f"vllm generated {self._total}, expected {tokens} tokens{ended}")
        return self.ttft_ms, (now - self.start) * 1000.0, self.itl_ms
