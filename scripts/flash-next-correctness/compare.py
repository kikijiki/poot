#!/usr/bin/env python3
"""POOT-1333: compare poot's next-token distributions with llama.cpp's, judged against llama.cpp's own noise.

Three distribution sources, all for the same token sequence, scored at the positions of the second half of
one `llama-perplexity` window (`first = n_ctx // 2 .. n_ctx - 2`; the record at position `p` is the distribution
of the token at `p + 1`):

  --cpu   a `llama-perplexity --save-all-logits` file from the CPU backend (the reference)
  --cuda  the same file from the CUDA backend (the reference's own noise: CPU vs CUDA on identical weights)
  --poot  `poot-score` output (top-K log-probabilities per position, from the production driver)

`compare.py tokens <logits file> <out>` writes the token ids of a logits file's window (the input of `poot-score`).

Everything is reduced to the candidate's top-K tokens (K = `--top`, the width poot-score records) before it is
compared, so llama.cpp against llama.cpp and llama.cpp against poot are measured the same way:

  kl        KL(reference || candidate) over the candidate's top-K tokens, both renormalized there
  top1      the two engines rank the same token first
  top5      fraction of the reference's five most likely tokens that the candidate's five also hold
  max_abs   largest |log-probability difference| of one token over the candidate's top-K (a log-probability
            differs from the logit by the same per-position constant in both engines, so this is the logit
            difference up to the log-sum-exp difference, which is what the card calls "max abs logit difference")

The tolerance is derived, not chosen: poot passes when each metric is within `--factor` times the CPU-vs-CUDA
noise (plus a small absolute floor so a noiseless pair does not demand exactness of poot) and its top-1 rate is
within `--top1-slack` of the noise pair's. The greedy continuations (`llama-server` responses
with `return_tokens`, and poot's) are compared by the index of the first differing token.

Exit status: 0 when poot is within tolerance, 1 when it is not, 2 for a malformed input.
"""

import argparse
import array
import heapq
import json
import math
import struct
import sys

MAGIC = b"_logits_"


class Malformed(Exception):
    pass


class Reference:
    """A llama.cpp `--save-all-logits` file (`tools/perplexity/perplexity.cpp`, `kl_divergence()`): the magic,
    `n_ctx`, `n_vocab`, `n_chunk` (32-bit), the tokens, then per chunk one record per scored position. A record is
    `2 * ceil(n_vocab / 2) + 4` 16-bit values: an f32 scale and an f32 floor, then one code per vocabulary entry;
    entry `i` has log-probability `scale * code[i] + floor`, and code 0 means "below the 16-nat clamp"."""

    def __init__(self, data):
        if data[: len(MAGIC)] != MAGIC:
            raise Malformed("not a llama.cpp logits file")
        at = len(MAGIC)
        try:
            self.n_ctx, self.n_vocab, self.chunks = struct.unpack_from("<iii", data, at)
        except struct.error:
            raise Malformed("truncated header")
        at += 12
        if min(self.n_ctx, self.n_vocab, self.chunks) < 1 or self.n_ctx < 3:
            raise Malformed("header %r" % ((self.n_ctx, self.n_vocab, self.chunks),))
        count = self.n_ctx * self.chunks
        self.tokens = list(struct.unpack_from("<%di" % count, data, at))
        at += 4 * count
        self.first = self.n_ctx // 2
        self.scored = self.n_ctx - 1 - self.first
        self.record_u16s = 2 * ((self.n_vocab + 1) // 2) + 4
        size = 2 * self.record_u16s * self.scored * self.chunks
        if len(data) < at + size:
            raise Malformed("the records are truncated")
        self.data = data
        self.at = at

    def window(self, chunk):
        return self.tokens[chunk * self.n_ctx : (chunk + 1) * self.n_ctx]

    def record(self, chunk, j):
        """(scale, floor, codes) of the `j`th scored position of `chunk`."""
        offset = self.at + 2 * self.record_u16s * (chunk * self.scored + j)
        scale, floor = struct.unpack_from("<ff", self.data, offset)
        codes = array.array("H")
        codes.frombytes(self.data[offset + 8 : offset + 8 + 2 * self.n_vocab])
        if sys.byteorder == "big":
            codes.byteswap()
        return scale, floor, codes

    def top(self, chunk, j, k):
        """The `k` most likely tokens of position `j` as {token: log-probability}, best first."""
        scale, floor, codes = self.record(chunk, j)
        best = heapq.nlargest(k, range(self.n_vocab), key=codes.__getitem__)
        return {t: scale * codes[t] + floor for t in best}

    def lookup(self, chunk, j, tokens):
        """The log-probability of each of `tokens`; None where the file holds only the clamp."""
        scale, floor, codes = self.record(chunk, j)
        return {t: (scale * codes[t] + floor if codes[t] else None) for t in tokens}


def load_reference(path):
    with open(path, "rb") as handle:
        return Reference(handle.read())


def reference_positions(reference, k):
    """[(position, {token: logprob} of the top k)] for chunk 0, and the window's tokens."""
    return [
        (reference.first + j, reference.top(0, j, k)) for j in range(reference.scored)
    ], reference.window(0)


def poot_positions(path):
    with open(path) as handle:
        report = json.load(handle)
    positions = {
        entry["pos"]: {int(t): float(lp) for t, lp in entry["top"]}
        for entry in report["positions"]
    }
    return positions, report


def renormalized(logprobs):
    mass = math.log(sum(math.exp(x) for x in logprobs.values()))
    return {t: x - mass for t, x in logprobs.items()}


def agree_at(ref, cand):
    """The metrics of one position. `ref` and `cand` are {token: logprob} over each side's top-K; `ref` must
    hold every token of `cand`'s support (the caller widened it), or the position is dropped from KL."""
    support = [t for t in cand if t in ref and ref[t] is not None]
    if not support:
        return None
    r = renormalized({t: ref[t] for t in support})
    c = renormalized({t: cand[t] for t in support})
    kl = sum(math.exp(r[t]) * (r[t] - c[t]) for t in support)
    max_abs = max(abs(ref[t] - cand[t]) for t in support)
    ref_order = sorted((t for t in ref if ref[t] is not None), key=lambda t: -ref[t])
    cand_order = sorted(cand, key=lambda t: -cand[t])
    return {
        "kl": kl,
        "max_abs": max_abs,
        "top1": ref_order[0] == cand_order[0],
        "top5": len(set(ref_order[:5]) & set(cand_order[:5])) / 5.0,
    }


def summarize(per_position, worst=5):
    rows = [(p, m) for p, m in per_position if m is not None]
    if not rows:
        raise Malformed("no position has a token in common")
    n = len(rows)
    return {
        "positions": n,
        "dropped": len(per_position) - n,
        "mean_kl": sum(m["kl"] for _, m in rows) / n,
        "max_kl": max(m["kl"] for _, m in rows),
        "max_abs": max(m["max_abs"] for _, m in rows),
        "top1": sum(m["top1"] for _, m in rows) / n,
        "top5": sum(m["top5"] for _, m in rows) / n,
        "worst": [
            {"pos": p, "kl": m["kl"], "max_abs": m["max_abs"]}
            for p, m in sorted(rows, key=lambda r: -r[1]["kl"])[:worst]
        ],
    }


def reference_vs_reference(ref, other, k):
    """Agreement of the llama.cpp file `other` with `ref`, with `other` cut to its top `k` first."""
    if (ref.n_ctx, ref.n_vocab, ref.window(0)) != (other.n_ctx, other.n_vocab, other.window(0)):
        raise Malformed("the two llama.cpp files score different windows")
    rows = []
    for j in range(ref.scored):
        cand = other.top(0, j, k)
        full = ref.lookup(0, j, cand)
        rows.append((ref.first + j, agree_at(ref_full(ref, j, full), cand)))
    return summarize(rows)


def ref_full(ref, j, looked_up):
    """The reference's distribution as the comparison reads it: its own top tokens plus the looked-up ones."""
    merged = dict(ref.top(0, j, 5))
    merged.update(looked_up)
    return {t: lp for t, lp in merged.items() if lp is not None}


def reference_vs_poot(ref, positions, k):
    tokens = ref.window(0)
    rows = []
    for j in range(ref.scored):
        pos = ref.first + j
        cand = positions.get(pos)
        if cand is None:
            raise Malformed("poot-score has no position %d" % pos)
        full = ref.lookup(0, j, cand)
        rows.append((pos, agree_at(ref_full(ref, j, full), cand)))
    return summarize(rows)


def first_divergence(a, b):
    """The index of the first element where `a` and `b` differ (the shorter length when one is a prefix of the
    other), or None when they are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def tolerance(noise, factor, top1_slack, floors):
    # The top-1 rate moves in steps of 1/positions: a window of two dozen positions cannot resolve 2%, and a
    # near-tie flipping at one or two positions is what the CPU-vs-CUDA pair does too.
    top1_slack = max(top1_slack, 2.0 / noise["positions"])
    return {
        "mean_kl": max(factor * noise["mean_kl"], floors["mean_kl"]),
        "max_kl": max(factor * noise["max_kl"], floors["max_kl"]),
        "max_abs": max(factor * noise["max_abs"], floors["max_abs"]),
        "top1": noise["top1"] - top1_slack,
    }


MAX_DROPPED = 0.1


def judge(subject, limits):
    """The failed gates of `subject` against `limits`: a list of (metric, observed, limit). A subject that lost
    more than MAX_DROPPED of its positions (no token in common with the reference) is a failure, not a smaller
    sample."""
    failures = []
    total = subject["positions"] + subject["dropped"]
    if subject["dropped"] > MAX_DROPPED * total:
        failures.append(("dropped", subject["dropped"] / total, MAX_DROPPED))
    for metric in ("mean_kl", "max_kl", "max_abs"):
        value = subject[metric]
        # A NaN compares false against everything: refuse it explicitly instead of passing it.
        if not value <= limits[metric]:
            failures.append((metric, value, limits[metric]))
    if not subject["top1"] >= limits["top1"]:
        failures.append(("top1", subject["top1"], limits["top1"]))
    return failures


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["tokens"]:
        try:
            ids = " ".join(map(str, load_reference(argv[1]).window(0)))
        except (Malformed, OSError) as error:
            print("malformed input: %s" % error, file=sys.stderr)
            return 2
        with open(argv[2], "w") as handle:
            handle.write(ids + "\n")
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cpu", required=True, help="llama-perplexity logits file, CPU backend (reference)")
    parser.add_argument("--cuda", required=True, help="the same, CUDA backend (the reference's noise)")
    parser.add_argument("--poot", required=True, help="poot-score JSON")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--factor", type=float, default=3.0, help="tolerance = factor x the CPU-vs-CUDA noise")
    parser.add_argument("--top1-slack", type=float, default=0.02)
    parser.add_argument("--floor-kl", type=float, default=1e-3)
    parser.add_argument("--floor-max-abs", type=float, default=0.05)
    parser.add_argument("--greedy-cpu", help="llama-server /completion response of the greedy continuation, CPU")
    parser.add_argument("--greedy-cuda", help="the same, CUDA")
    parser.add_argument("--out", help="write the report JSON here")
    args = parser.parse_args(argv)

    try:
        cpu, cuda = load_reference(args.cpu), load_reference(args.cuda)
        positions, poot = poot_positions(args.poot)
        noise = reference_vs_reference(cpu, cuda, args.top)
        subject = reference_vs_poot(cpu, positions, args.top)
        subject_vs_cuda = reference_vs_poot(cuda, positions, args.top)
    except Malformed as error:
        print("malformed input: %s" % error, file=sys.stderr)
        return 2

    floors = {"mean_kl": args.floor_kl, "max_kl": 10 * args.floor_kl, "max_abs": args.floor_max_abs}
    limits = tolerance(noise, args.factor, args.top1_slack, floors)
    failures = judge(subject, limits)

    report = {
        "top": args.top,
        "window_tokens": cpu.n_ctx,
        "scored_positions": cpu.scored,
        "poot_device": poot.get("adapter"),
        "prefill_chunk": poot.get("prefill_chunk"),
        "noise_cpu_vs_cuda": noise,
        "poot_vs_cpu": subject,
        "poot_vs_cuda": subject_vs_cuda,
        "limits": limits,
        "failures": [{"metric": m, "observed": v, "limit": l} for m, v, l in failures],
        "greedy": greedy_report(args, poot),
        "pass": not failures,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return 0 if not failures else 1


def greedy_report(args, poot):
    runs = {"poot": poot["greedy"]["ids"]}
    for name, path in (("llama_cpu", args.greedy_cpu), ("llama_cuda", args.greedy_cuda)):
        if path:
            with open(path) as handle:
                runs[name] = json.load(handle)["tokens"]
    report = {"poot_text": poot["greedy"]["text"]}
    names = list(runs)
    if "llama_cpu" in runs and "llama_cuda" in runs:
        # Informational, not a gate: one 64-token sample cannot carry a verdict. poot is "within noise" when it
        # follows the CPU reference for at least as long as the CUDA backend does.
        inf = float("inf")
        poot_at = first_divergence(runs["llama_cpu"], runs["poot"])
        noise_at = first_divergence(runs["llama_cpu"], runs["llama_cuda"])
        report["poot_diverges_no_earlier_than_llama_cuda"] = (inf if poot_at is None else poot_at) >= (
            inf if noise_at is None else noise_at
        )
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            report["%s_vs_%s" % (a, b)] = {
                "first_divergence_token": first_divergence(runs[a], runs[b]),
                "tokens": [len(runs[a]), len(runs[b])],
            }
    return report


if __name__ == "__main__":
    sys.exit(main())
