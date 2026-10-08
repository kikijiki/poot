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
cargo run --release -p poot-llm --features cli --bin poot-caption -- /path/to/smolvlm image.jpg "Describe this image." --backend wgpu
```

It loads the checkpoint as two registered families, the vision tower and the text decoder, each on its own driver
(`--backend` is `wgpu`, `rocm`, `ptx`, `vulkan` or `auto`, the default). The image is split into frames, each frame
runs through the vision tower, and the text decoder generates with the frames' embedding rows scattered over the
prompt's `<image>` placeholders. The library path is `poot_llm::front_end::Idefics3`.
