---
id: multimodal
title: Multimodal
sidebar_position: 8
---

# Multimodal

`poot-serve` takes image input on `/v1/chat/completions` when the served checkpoint's family has a vision front end (see [Vision-language](../develop/models.md#vision-language)), and answers with the text the decoder generates from the images.

## Request shape

A request with images can hold a whole conversation: system, user and assistant messages in any order, with `image_url` parts in any message other than a tool message. The conversation is rendered through the checkpoint's own chat template, a message's images before its text. The `image_url` parts are `data:` URIs with base64 PNG or JPEG content; the server fetches no URLs. `stop`, `response_format` and the guided fields apply as usual.

## Limits and refusals

Each of these is a `400`:

- a URL instead of a `data:` URI, or an image that does not decode;
- a message text that spells an image placeholder;
- more than 16 images in one request (refused before any base64 is decoded);
- images whose headers declare more than 64 Mi pixels together (refused before any pixel is decoded);
- `tools`, `tool_choice`, a tool call or a tool message (the image chat template has no tool branch);
- an image or frame count above what the checkpoint's vision front end accepts.

A checkpoint without a vision front end refuses an image part with a `400` `unsupported_capability` error, as does a video or audio part on any checkpoint. A chat template that prints a different number of placeholders than the request has images is a server error, not a `400`.

## Timeouts and health

Images are decoded and encoded on a vision thread, off the engine thread. A request waits at most five minutes for it and stops waiting when its client hangs up. A panic of that thread turns `GET /health` to `dead` and fails the requests waiting on it with a `500`.

## Captioning from the command line

The `poot caption` command of `poot-llm` captions an image without the server:

```bash
cargo run --release -p poot-llm --features cli --bin poot -- \
  caption <path-to-checkpoint> <image> "Describe this image." --backend wgpu
```

`--backend` is `wgpu`, `rocm`, `ptx`, `vulkan` or `auto` (the default). The library path is `ModelHandle::load` and `poot_llm::FrontEndSession`.
