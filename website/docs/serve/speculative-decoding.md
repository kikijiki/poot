---
id: speculative-decoding
title: Speculative decoding
sidebar_position: 10
---

# Speculative decoding

Speculative decoding is a driver policy (`DriverOptions::draft`). A decode row under a policy returns one to `k + 1` tokens a step instead of one. The tokens are those the request would have without the policy; only the number of steps changes.

## Serving it

`poot-serve` takes the policy on the command line. The three flags exclude each other; without one, a decode row is one token.

| Flag                  | Proposes                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------- |
| `--draft-lookup NGRAM:K` | Up to `K` tokens that followed the latest earlier occurrence of the last `NGRAM` tokens. No second model. |
| `--draft-model PATH:K`   | `K` tokens a round from a second, smaller model loaded from `PATH`.                    |
| `--draft-mtp K`          | `K` tokens a round from the served model's own multi-token-prediction layer. `K` is required; start at one and measure two or three. |

A draft path that does not load, or a draft the driver refuses, fails startup before the listener binds. `/metrics` reports `speculative_rounds`, `speculative_proposed_tokens`, `speculative_accepted_tokens` (drafts kept, not tokens returned) and `speculative_skipped_rows`, also exported as `poot_speculative_*_total` counters.

## What a round guarantees

The driver verifies the drafts in one step with the logits of every position. Every model that supports chunked prefill verifies.

- **Greedy rows** keep the drafts that equal the target's argmax, then the target's own next token. The tokens are exactly those of a run without the policy.
- **Sampled rows** accept a draft with probability `min(1, p/q)` and resample a rejection from the normalized `max(0, p - q)`, so the output distribution is the target's.
- **Stops:** a stop string, the end-of-sequence token or the token limit inside a multi-token result cuts the result there.

A model with recurrent state restores its state after a rejected draft and replays the accepted tokens. A rejecting round costs one more forward, and each row holds one extra copy of the state. Measure with the policy and prompts you intend to serve: these models, and draft models, gain only where drafts are accepted often enough to pay for it.

## When a row does not speculate

A row decodes one token when:

- its sampler has a penalty, a logit bias, a constraint or logprob recording;
- the sequence is too near its capacity to hold a verify window;
- the sequence carries request inputs the draft cannot read (embedding rows, sectioned positions);
- the policy proposed nothing (no earlier match for prompt lookup; no MTP draft after a partly committed result).

Each case is counted in `speculative_skipped_rows`; `Driver::stats().speculation` breaks the count down by reason.

## What is refused

`Driver::prepare` returns a typed `Unsupported::Speculation` or `Unsupported::Incompatible` error, and `poot-serve` stops at startup with it:

- a model whose recurrent state advances in granules of more than one token;
- a draft model with recurrent state, a larger vocabulary than the target's, a tokenizer that gives a token another id, or fewer positions than the driver's capacity;
- `--draft-mtp` on a model with no MTP program, or an MTP program with recurrent state, with an adapter set, with beam search, or over a device group (`--placement pipeline:` or `tensor:`).

Under `--draft-mtp` prefix reuse is off: a resumed conversation reuses one position less than without the policy. A prompt that carries request inputs such as embedding rows is admitted but decodes one token a step.

## Measuring a policy

The `speculation_bench` example of `poot-llm` runs one policy on one checkpoint and prompt. It prints the decode rate, the acceptance rate (overall and per draft position), the share of rounds that rejected a draft, and a digest of the generated tokens.

- `--timing 1` adds the engine's time per verified round; take the decode rate from a run without it.
- `speculation_prompts/` holds a mixed-text and a repetitive-text prompt of about 570 tokens.
- Run it once per policy on the same binary and prompt. A greedy run must print the digest the policy-off run prints.

Design: [Serving design](../architecture/serving-design.mdx).
