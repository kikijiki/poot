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
- `--draft-mtp K` serves the model's own multi-token-prediction layer, `K` tokens a round. `K` is required: there is no default. One is the usual start for GLM, and two or three are worth measuring on your prompts, since a single reused layer drafts worse the deeper it chains. A model with no such layer is refused at startup.

The three flags exclude each other. Without any, a decode row is one token. A request's tokens are those it would have without the policy; the policy changes only how many steps they take. `/metrics` reports what the driver counted for every decode row, as the driver reported it: `speculative_rounds`, `speculative_proposed_tokens`, `speculative_accepted_tokens` (drafts kept, not tokens returned: a round returns its accepted drafts and one more) and `speculative_skipped_rows`, also exported as `poot_speculative_*_total` counters.

Speculative decoding is a policy of the `poot-llm` driver, not a decoding loop of its own and not a list of models. A driver is built with a `DraftPolicy` in `DriverOptions::draft`, and a decode row of `Driver::step` then returns one to `k + 1` tokens instead of one.

## Policies

- `DraftPolicy::Off`: a decode row is one token.
- `DraftPolicy::PromptLookup { ngram, k }`: propose the tokens that followed the most recent earlier occurrence of the sequence's last `ngram` tokens, up to `k`. It needs no second model.
- `DraftPolicy::DraftModel { handle, k }`: propose `k` tokens with a second, smaller model. It loads as a second executable on the same executor, and it keeps its own cache in step with the target's through the same `open`, `commit` and `release`.
- `DraftPolicy::Mtp { k }`: propose `k` tokens with the served model's own multi-token-prediction layer. It needs no second model and no second executable: its program runs on the target's executable, over the hidden state the target's programs leave at every position they run. `k` has no default.

## How a round runs

The driver proposes drafts, then verifies them in one step: the sequence's last committed token followed by the drafts, with the logits of every position. That step is the model's ordinary prefill program at a nonzero start, so every model that supports chunked prefill verifies; no list of models decides it. `Driver::prepare` compiles one verify entry per row count beside decode, so a round with fewer than `k` proposals reuses it and nothing compiles while serving.

- Greedy rows keep the drafts that equal the target's argmax, then the target's own next token. The tokens are exactly those of `DraftPolicy::Off`.
- Sampled rows use speculative sampling over the target's and the draft's distributions: a draft is accepted with probability `min(1, p/q)`, and a rejection resamples from the normalized `max(0, p - q)`. The output distribution is the target's. For prompt lookup the draft's distribution is the one-hot row of the proposed token.

Under `DraftPolicy::Mtp` the layer's program runs after every step, over the positions that step wrote, so its cache holds every committed position and the last one yields the first draft of the sequence's next round; the other `k - 1` drafts are one-token steps at the start of the round, each fed the draft before it and the state the step before left. The layer keeps its own cache, which follows the committed tokens: a round that rejects a draft is followed by a catch-up over the accepted ones, and a result committed in part leaves the sequence with no draft for its next round. Each draft is a point mass on the layer's own pick, whatever the row's sampler, so a sampled row verifies by `u < p(d)` as prompt lookup does and the output distribution stays the target's. A model with recurrent state (GLM-5.3 has KDA layers) gets the same shadow-set rollback as above for the target; the layer itself must hold positional state only.

`commit` keeps the first `kept` tokens of a result, so a stop string, the end-of-sequence token or the token limit inside a multi-token result needs no rollback: the cache positions past the kept tokens hold stale entries that no later step reads.

A model with recurrent state (Qwen3-Next, Nemotron-H) needs a rollback of its own, because a rejected draft token has already been folded into that state. Before the verify step the driver copies the recurrent state of the rows that hold drafts into a shadow set. A row that keeps every draft keeps its state. A row that rejects one restores the copy and replays the sequence's last committed token and the accepted drafts as one chunk, so its state is exactly what decoding those tokens leaves. The verify window is as wide as the row's drafts, since padding would fold into the state. Positional state needs none of this. The shadow set costs one extra copy of the recurrent state per row, and a rejecting round costs one more forward over the kept tokens, so these models gain only where drafts are accepted often enough to pay for it; measure with the policy and prompts you intend to serve. A result of such a row is kept whole or its sequence ends, since the state already holds the kept tokens: a shorter commit followed by another step on the sequence is refused with a typed `RecurrentStateAhead` error.

## When a row does not speculate

A decode row under a draft policy decodes one token, with the same output it would have without the policy, in three cases. The row says which in `RowResult::speculation`:

- `NotSpeculated::Sampler`: the row's sampler has a penalty, a logit bias, a constraint or logprob recording. The window's acceptance does not reproduce those draws.
- `NotSpeculated::NoRoom`: the sequence is too near its capacity to hold a verify window.
- `NotSpeculated::Inputs`: the sequence carries request inputs (embedding rows, sectioned positions) the draft cannot read, so it is not speculated.
- `NotSpeculated::NoProposal`: the policy proposed nothing (prompt lookup found no earlier match; the MTP layer has no draft because the sequence's last result was committed in part).

A row that verified drafts reports `Verified { proposed, accepted }`; a prefill row, a row with a caller-supplied `ahead` window, and every row under `Off` report `NotApplicable`. `Driver::stats().speculation` sums these over the driver's life: rounds, proposed and accepted drafts (the acceptance rate is accepted over proposed), and one counter per `NotSpeculated` reason.

## Measuring a policy

The `speculation_bench` example of `poot-llm` runs one policy on one checkpoint and prompt and prints the decode rate, the acceptance rate (overall and per draft position), `p` (the share of verified rounds that rejected a draft, so restored and replayed a recurrent target) and a digest of the generated tokens. With `--timing 1` it adds the engine's time per verified round for the draft passes, the verify window and the snapshot, restore and replay copies, and the host remainder; take the decode rate from a run without it. `speculation_prompts/` holds a mixed-text and a repetitive-text prompt of about 570 tokens. Run it once per policy on the same binary and prompt: a greedy run must print the digest `off` prints, and the acceptance rate is the driver's own count. A draft model gives a gain only when it agrees with the target often enough to pay for its own steps, so measure with the draft you intend to serve.

## What is refused

`Driver::prepare` returns a typed `Unsupported::Speculation` or `Unsupported::Incompatible` error, before anything compiles or uploads where it can. `Incompatible` names the two features that cannot be combined and the reason, from the one table the driver checks every combination against:

- `RecurrentGranule { granule }`: the model's recurrent state advances in granules of more than one token. The replay chunk after a rejection is not a multiple of the granule, so these models do not speculate. The check reads the state roles of the compiled program and the model's granule, never a model name.
- `Incompatible` draft model and recurrent draft state: the draft model carries recurrent state. Only the target's state is rolled back, so a rejected draft token would stay folded into the draft's.
- `DraftMismatch`: the draft model's vocabulary is larger than the target's, its tokenizer gives a token another id, or it holds fewer positions than the driver's capacity.
- `Incompatible` MTP and recurrent draft state: the model's MTP program carries recurrent state. The rollback's snapshot schema is the target's decode program, so state only the layer holds would never be restored after a rejected draft.
- `Incompatible` MTP and adapters: an adapter set was named under `--draft-mtp`. The layer drafts for the base weights; an adapter changes what the target computes.
- `Incompatible` MTP and beam search: beam search under `--draft-mtp`. A beam runs the target's programs without the hidden-state carry the layer reads.

Under `--draft-mtp` a model whose family traces no MTP program is refused with the family's own typed program refusal, a device group (`--placement pipeline:` or `tensor:`) is refused with the placement named, a prompt that carries request inputs such as embedding rows is admitted but not speculated (the layer would embed the placeholder tokens the rows replace), so its rows decode one token a step, and prefix reuse is turned off: the layer's cache at a block's last position depends on the token after the block, so a finished sequence registers no blocks and a later prompt reuses none. A resumed conversation reuses one position less than without the policy.

`poot-serve` reports the same refusals at startup, before the listener binds: a `--draft-model` that does not load or does not fit, a draft flag on a model with a recurrent granule above one token, or `--draft-mtp` on a model with no MTP layer, stops the server with the driver's typed error. A request whose sampler the verify window cannot check (a penalty, `logit_bias`, logprobs or a constraint) decodes one token a step and is counted in `speculative_skipped_rows`.

Design notes on the method live under [Architecture: Serving design](../architecture/serving-design.mdx).
