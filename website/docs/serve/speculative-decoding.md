---
id: speculative-decoding
title: Speculative decoding
sidebar_position: 10
---

# Speculative decoding

## Serving it

`poot-serve` takes the policy on the command line and hands it to the driver; the one scheduling loop does not change.

- `--draft-lookup NGRAM:K` serves prompt lookup: the last `NGRAM` tokens of a sequence are matched against its earlier text and up to `K` following tokens are proposed.
- `--draft-model PATH:K` serves a draft model loaded from `PATH`, `K` tokens a round. A path that does not load, or a draft the driver refuses (see below), fails startup before the listener binds.

The two flags exclude each other. Without either, a decode row is one token. A request's tokens are those it would have without the policy; the policy changes only how many steps they take. `/metrics` reports what the driver counted for every decode row, as the driver reported it: `speculative_rounds`, `speculative_proposed_tokens`, `speculative_accepted_tokens` (drafts kept, not tokens returned: a round returns its accepted drafts and one more) and `speculative_skipped_rows`, also exported as `poot_speculative_*_total` counters.

Speculative decoding is a policy of the `poot-llm` driver, not a decoding loop of its own and not a list of models. A driver is built with a `DraftPolicy` in `DriverOptions::draft`, and a decode row of `Driver::step` then returns one to `k + 1` tokens instead of one.

## Policies

- `DraftPolicy::Off`: a decode row is one token.
- `DraftPolicy::PromptLookup { ngram, k }`: propose the tokens that followed the most recent earlier occurrence of the sequence's last `ngram` tokens, up to `k`. It needs no second model.
- `DraftPolicy::DraftModel { handle, k }`: propose `k` tokens with a second, smaller model. It loads as a second executable on the same executor, and it keeps its own cache in step with the target's through the same `open`, `commit` and `release`.

## How a round runs

The driver proposes drafts, then verifies them in one step: the sequence's last committed token followed by the drafts, with the logits of every position. That step is the model's ordinary prefill program at a nonzero start, so every model that supports chunked prefill verifies; no list of models decides it. `Driver::prepare` compiles one verify entry per row count beside decode, so a round with fewer than `k` proposals reuses it and nothing compiles while serving.

- Greedy rows keep the drafts that equal the target's argmax, then the target's own next token. The tokens are exactly those of `DraftPolicy::Off`.
- Sampled rows use speculative sampling over the target's and the draft's distributions: a draft is accepted with probability `min(1, p/q)`, and a rejection resamples from the normalized `max(0, p - q)`. The output distribution is the target's. For prompt lookup the draft's distribution is the one-hot row of the proposed token.

`commit` keeps the first `kept` tokens of a result, so a stop string, the end-of-sequence token or the token limit inside a multi-token result needs no rollback: the cache positions past the kept tokens hold stale entries that no later step reads.

## When a row does not speculate

A decode row under a draft policy decodes one token, with the same output it would have without the policy, in three cases. The row says which in `RowResult::speculation`:

- `NotSpeculated::Sampler`: the row's sampler has a penalty, a logit bias, a constraint or logprob recording. The window's acceptance does not reproduce those draws.
- `NotSpeculated::NoRoom`: the sequence is too near its capacity to hold a verify window.
- `NotSpeculated::NoProposal`: the policy proposed nothing (prompt lookup found no earlier match).

A row that verified drafts reports `Verified { proposed, accepted }`; a prefill row, a row with a caller-supplied `ahead` window, and every row under `Off` report `NotApplicable`. `Driver::stats().speculation` sums these over the driver's life: rounds, proposed and accepted drafts (the acceptance rate is accepted over proposed), and one counter per `NotSpeculated` reason.

## Measuring a policy

The `speculation_bench` example of `poot-llm` runs one policy on one checkpoint and prompt and prints the decode rate, the acceptance rate and a digest of the generated tokens. Run it once per policy (`off`, `lookup`, `model`) on the same binary and prompt: a greedy run must print the digest `off` prints, and the acceptance rate is the driver's own count. A draft model gives a gain only when it agrees with the target often enough to pay for its own steps, so measure with the draft you intend to serve.

## What is refused

`Driver::prepare` returns a typed `Unsupported::Speculation` error, before anything compiles or uploads where it can:

- `RecurrentState`: the model carries recurrent state (Qwen3-Next, Nemotron-H, Mamba2). A rejected draft token has already been folded into that state and `commit` cannot undo it, so these models do not speculate. The check reads the state roles of the compiled program, never a model name.
- `DraftMismatch`: the draft model's vocabulary is larger than the target's, its tokenizer gives a token another id, or it holds fewer positions than the driver's capacity.

`poot-serve` reports the same refusals at startup, before the listener binds: a `--draft-model` that does not load or does not fit, or a draft flag on a model with recurrent state, stops the server with the driver's typed error. A request whose sampler the verify window cannot check (a penalty, `logit_bias`, logprobs or a constraint) decodes one token a step and is counted in `speculative_skipped_rows`.

Design notes on the method live under [Architecture: Serving design](../architecture/serving-design.mdx).
