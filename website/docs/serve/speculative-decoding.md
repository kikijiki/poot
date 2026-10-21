---
id: speculative-decoding
title: Speculative decoding
sidebar_position: 10
---

# Speculative decoding

Speculative decoding is a policy of the `poot-llm` driver, set in `DriverOptions::draft`, not a decoding loop of its
own. A decode row under a policy returns one to `k + 1` tokens a step instead of one. A request's tokens are those it
would have without the policy; the policy changes only how many steps they take.

## Serving it

`poot-serve` takes the policy on the command line; the one scheduling loop does not change.

- `--draft-lookup NGRAM:K` proposes prompt lookup: the tokens that followed the most recent earlier occurrence of a
  sequence's last `NGRAM` tokens, up to `K`. It needs no second model.
- `--draft-model PATH:K` proposes `K` tokens a round with a second, smaller model loaded from `PATH`. A path that
  does not load, or a draft the driver refuses (see below), fails startup before the listener binds.
- `--draft-mtp K` proposes `K` tokens a round with the served model's own multi-token-prediction layer. `K` is
  required: there is no default. One is the usual start, and two or three are worth measuring on your prompts, since
  a single reused layer drafts worse the deeper it chains. A model with no such layer is refused at startup.

The three flags exclude each other. Without any, a decode row is one token. `/metrics` reports what the driver
counted: `speculative_rounds`, `speculative_proposed_tokens`, `speculative_accepted_tokens` (drafts kept, not tokens
returned: a round returns its accepted drafts and one more) and `speculative_skipped_rows`, also exported as
`poot_speculative_*_total` counters.

## What a round guarantees

The driver verifies the drafts in one step with the logits of every position. Every model that supports chunked
prefill verifies; no list of models decides it, and nothing compiles while serving.

- Greedy rows keep the drafts that equal the target's argmax, then the target's own next token. The tokens are
  exactly those of a run without the policy.
- Sampled rows use speculative sampling: a draft is accepted with probability `min(1, p/q)` and a rejection
  resamples from the normalized `max(0, p - q)`, so the output distribution is the target's. Prompt-lookup and MTP
  drafts are point masses on the proposed token.
- A stop string, the end-of-sequence token or the token limit inside a multi-token result cuts the result there.

A model with recurrent state restores its state after a rejected draft and replays the accepted tokens, so a
rejecting round costs one more forward and each row holds one extra copy of the recurrent state. These models gain
only where drafts are accepted often enough to pay for it; measure with the policy and prompts you intend to serve. A
draft model likewise gains only when it agrees with the target often enough to pay for its own steps.

## When a row does not speculate

A decode row decodes one token, with the output it would have without the policy, when:

- its sampler has a penalty, a logit bias, a constraint or logprob recording;
- the sequence is too near its capacity to hold a verify window;
- the sequence carries request inputs the draft cannot read (embedding rows, sectioned positions);
- the policy proposed nothing (no earlier match for prompt lookup; no MTP draft because the sequence's last result
  was committed in part).

Each case is counted in `speculative_skipped_rows`; `Driver::stats().speculation` breaks the count down by reason.

## What is refused

`Driver::prepare` returns a typed `Unsupported::Speculation` or `Unsupported::Incompatible` error before anything
compiles or uploads where it can, and `poot-serve` stops at startup with that error:

- a model whose recurrent state advances in granules of more than one token cannot speculate;
- a draft model with recurrent state, or one whose vocabulary is larger than the target's, whose tokenizer gives a
  token another id, or that holds fewer positions than the driver's capacity;
- `--draft-mtp` on a model whose family has no MTP program, or whose MTP program carries recurrent state, with an
  adapter set, with beam search, or over a device group (`--placement pipeline:` or `tensor:`).

Under `--draft-mtp` prefix reuse is off: a finished sequence registers no blocks and a later prompt reuses none, so a
resumed conversation reuses one position less than without the policy. A prompt that carries request inputs such as
embedding rows is admitted but decodes one token a step.

## Measuring a policy

The `speculation_bench` example of `poot-llm` runs one policy on one checkpoint and prompt and prints the decode
rate, the acceptance rate (overall and per draft position), the share of verified rounds that rejected a draft, and a
digest of the generated tokens. With `--timing 1` it adds the engine's time per verified round; take the decode rate
from a run without it. `speculation_prompts/` holds a mixed-text and a repetitive-text prompt of about 570 tokens.
Run it once per policy on the same binary and prompt: a greedy run must print the digest the policy-off run prints.

Design: [Architecture: Serving design](../architecture/serving-design.mdx).
