---
id: multimodal
title: Multimodal
sidebar_position: 8
---

# Multimodal

`poot-serve` serves a SmolVLM (idefics3) or a Qwen2.5-VL checkpoint: `/v1/chat/completions` takes `image_url`
content parts and answers with the text the decoder generates from the image. Other vision-language families are
not served yet.

The server loads the checkpoint as two registered families over one set of weights. The text decoder runs on the
engine's driver as for any model, with its embedding-row entries prepared. The vision tower runs on its own driver
and thread, opened before the engine so the engine fits its workload to the memory the tower left. A request's
images are decoded and encoded there, off the engine thread (SmolVLM splits them into frames first), and the
prompt's image placeholders then take the tower's rows. The sequence opens with those rows; no prefix is matched
or registered for it. For Qwen2.5-VL the request also carries the prompt's rotary position table: the image's
tokens sit at their own temporal, height and width positions, and the sequence opens with that table and continues
from where it ends.

A request with images can hold a whole conversation: system, user and assistant messages in any order, with
`image_url` parts in any message other than a tool message. The conversation is rendered through the checkpoint's
own chat template, with each image where its message carries it (a message's images come before its text), and each
image's placeholders then take its frames' rows. The `image_url` parts are `data:` URIs with base64 PNG or JPEG
content. The server fetches no URLs, so a URL, an image that does not decode, a message text that spells
`<image>`, `tools`, `tool_choice`, a tool call or a tool message (the image chat template has no tool branch), too many frames (SmolVLM), or an image over the tower's patch limit (Qwen2.5-VL) is a `400`. `stop`, `response_format` and the guided fields apply as usual. A checkpoint without an
image front end refuses an image part with a `400` `unsupported_capability` error, and so does a video or audio part
on any checkpoint, instead of dropping the part.

The `poot-caption` binary of `poot-llm` captions an image from the command line and does not use the
server:

```bash
cargo run --release -p poot-llm --features cli --bin poot-caption -- /path/to/smolvlm image.jpg "Describe this image." --backend wgpu
```

It loads the checkpoint as two registered families, the vision tower and the text decoder, each on its own driver
(`--backend` is `wgpu`, `rocm`, `ptx`, `vulkan` or `auto`, the default). The image is split into frames, each frame
runs through the vision tower, and the text decoder generates with the frames' embedding rows scattered over the
prompt's `<image>` placeholders. The library path is `poot_llm::front_end::Idefics3`.

A Qwen2.5-VL checkpoint takes the same command. Its vision tower (windowed attention over the image's patches) runs
on one driver and returns the rows the chat template's `<|image_pad|>` placeholders stand for; its text decoder runs
on the other with the prompt's image tokens at their own temporal, height and width rotary positions and generated
tokens continuing after them. An animated PNG is read as a video, sampled at two frames per second. The library path
is `poot_llm::qwen_vl::Qwen25Vl`; images are limited to 4096 patches (about 800,000 pixels), since the tower's
windows are a mask over dense attention; a larger image is refused with a typed error. The driver prepares one
vision program per distinct patch count, so a stream of differently sized images compiles one per size.
