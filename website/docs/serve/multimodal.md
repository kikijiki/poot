---
id: multimodal
title: Multimodal
sidebar_position: 8
---

# Multimodal

:::caution[Being restored]
poot is being refactored. This is not available yet and is planned to return.
:::

Today `poot-serve` serves no vision-language checkpoint.

A text-only decoder has no image, video or audio front end, so `/v1/chat/completions` refuses a request
that carries such a content part with a `400` `unsupported_capability` error naming the modality, instead
of dropping the part. A valid request returns `503` naming the missing engine.

The `poot-caption` binary of `poot-llm` captions an image from the command line and does not use the
server:

```bash
cargo run --release -p poot-llm --features cli --bin poot-caption -- /path/to/smolvlm image.jpg "Describe this image."
```
