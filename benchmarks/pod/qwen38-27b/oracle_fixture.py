"""POOT-1184: turn the qwen35_27b_fp8_receipt row's RECEIPT lines into the committed oracle fixture (JSON on stdout).

The row (crates/poot-llm/src/driver/device_tests.rs, generates_and_matches_the_oracle) prints, on stderr:
  RECEIPT prompt "<text>" ids [..]
  RECEIPT generated [..] text "<text>"
  RECEIPT device first-token top5 [(id, logprob), ..]
  RECEIPT oracle first-token top5 [(id, logprob), ..]
The oracle line is the CPU oracle over the same traced graph and weight map (f64 log-softmax of the prompt's last
position). Top-5 is all the row prints; a top-50 / full-logit dump needs a change to that test (not in this card's
Owns). Usage: python -I oracle_fixture.py <nextest-output-file> <backend> [checkpoint-revision]
"""

import json
import re
import sys

PAIRS = re.compile(r"\((\d+),\s*(-?[0-9.e+-]+|NaN|-?inf)\)")


def line(text, key):
    found = [l for l in text.splitlines() if l.lstrip().startswith("RECEIPT " + key)]
    if len(found) != 1:
        raise SystemExit(f"expected exactly one 'RECEIPT {key}' line, found {len(found)}")
    return found[0].split("RECEIPT " + key, 1)[1].strip()


def top(text, key):
    pairs = [(int(i), float(p)) for i, p in PAIRS.findall(line(text, key))]
    if not pairs:
        raise SystemExit(f"no (id, logprob) pairs in 'RECEIPT {key}'")
    return {"ids": [i for i, _ in pairs], "logprobs": [p for _, p in pairs]}


def main(path, backend, revision):
    text = open(path, encoding="utf-8", errors="replace").read()
    prompt = re.match(r'(".*")\s+ids\s+(\[[^\]]*\])', line(text, "prompt"))
    generated = re.match(r'(\[[^\]]*\])\s+text\s+(".*")', line(text, "generated"))
    if not prompt or not generated:
        raise SystemExit("unparsable RECEIPT prompt/generated line")
    print(json.dumps({
        "checkpoint": "Qwen/Qwen3.8-27B-FP8",
        "checkpoint_revision": revision,
        "source": "poot-llm device_tests::qwen35_27b_fp8_receipt (CPU oracle, f64 log-softmax of the last prompt position)",
        "device_backend": backend,
        "prompt": json.loads(prompt.group(1)),
        "prompt_ids": json.loads(prompt.group(2)),
        "generated_ids": json.loads(generated.group(1)),
        "generated_text": json.loads(generated.group(2)),
        "oracle_first_token_top5": top(text, "oracle first-token top5"),
        "device_first_token_top5": top(text, "device first-token top5"),
        "tolerance": "token ids exact, log-probabilities within 1e-3 (ADR-0101 tier 2)",
        "not_included": "top-50 / full logits: the row prints top-5 only",
    }, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "unknown")
