#!/usr/bin/env python3
"""KV cache dtype drift: two `poot-score` reports of one model, one sequence and one backend that differ only in
`--kv-dtype`, the first the reference (f32, the exact cache), the second the candidate.

    compare.py <reference.json> <candidate.json> [--json]

Per scored position (teacher-forced, so both runs see the same context) it reads the top-K log-probabilities each
run recorded and reports:

- top1: the fraction of positions whose most probable token agrees;
- dlp: |logprob_ref(t) - logprob_cand(t)| of the reference's top token t (mean, p99, max), the logit drift in the
  units a sampler sees; a token absent from the candidate's top-K counts at the candidate's K-th logprob, a
  lower bound;
- kl: KL(ref || cand) over the reference's top-K tokens, both renormalized over them (mean, max), with the same
  lower bound for absent tokens.

And for the greedy continuation (free-running, so one early flip changes the rest): the first index at which the
tokens differ and the fraction of equal tokens.
"""

import json
import math
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def top_k_kl(lp_ref, lp_cand):
    """KL(p || q) of the two distributions renormalized over the same tokens (so it is never negative)."""
    p = [math.exp(lp) for lp in lp_ref]
    q = [math.exp(lp) for lp in lp_cand]
    sp, sq = sum(p), sum(q)
    return sum(pi / sp * math.log((pi / sp) / (qi / sq)) for pi, qi in zip(p, q))


def compare(ref, cand):
    for field in ("tokens", "start"):
        if ref[field] != cand[field]:
            raise SystemExit("the reports score different sequences (%s %r != %r)" % (field, ref[field], cand[field]))
    cand_at = {p["pos"]: p["top"] for p in cand["positions"]}
    agree, dlps, kls = 0, [], []
    for position in ref["positions"]:
        top_ref = position["top"]
        top_cand = cand_at[position["pos"]]
        lp_cand = {token: lp for token, lp in top_cand}
        floor = min(lp for _, lp in top_cand)
        if top_ref[0][0] == top_cand[0][0]:
            agree += 1
        best, lp_best = top_ref[0]
        dlps.append(abs(lp_best - lp_cand.get(best, floor)))
        kls.append(top_k_kl([lp for _, lp in top_ref], [lp_cand.get(token, floor) for token, _ in top_ref]))
    greedy_ref, greedy_cand = ref["greedy"]["ids"], cand["greedy"]["ids"]
    pairs = list(zip(greedy_ref, greedy_cand))
    first = next((i for i, (a, b) in enumerate(pairs) if a != b), None)
    n = len(ref["positions"])
    return {
        "reference_kv": ref.get("kv_dtype"),
        "candidate_kv": cand.get("kv_dtype"),
        "positions": n,
        "top1_agree": agree / n,
        "dlp_mean": sum(dlps) / n,
        "dlp_p99": quantile(dlps, 0.99),
        "dlp_max": max(dlps),
        "kl_mean": sum(kls) / n,
        "kl_max": max(kls),
        "greedy_tokens": len(pairs),
        "greedy_first_divergence": first,
        "greedy_equal": sum(a == b for a, b in pairs) / max(1, len(pairs)),
    }


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2:
        raise SystemExit(__doc__)
    result = compare(load(args[0]), load(args[1]))
    if "--json" in argv:
        print(json.dumps(result))
        return
    for key, value in result.items():
        print("%-24s %s" % (key, "%.6g" % value if isinstance(value, float) else value))


if __name__ == "__main__":
    main(sys.argv[1:])
