---
id: lora-adapters
title: Using LoRA adapters
sidebar_position: 11
---

# Using LoRA adapters

:::caution[Being restored]
poot is being refactored. This is not available yet and is planned to return.
:::

Today registering an adapter is refused, at startup (`--lora-adapter`) and by hot-load (`POST /v1/lora_adapters`).

## What the server does today

- `--lora-adapter NAME=DIR` and `--lora-pool-capacity N` are still parsed, so a startup command that
  passes them fails with the registration refusal instead of an unknown-option error.
- `GET /v1/lora_adapters` lists no adapters. `POST /v1/lora_adapters` is refused, and
  `DELETE /v1/lora_adapters/NAME` finds no registered adapter to unload.
- A completion request that names a `lora_adapter` is a `400` error (no adapter is registered).
- The administration routes keep their trust policy: on a non-loopback listener they are disabled unless
  `POOT_LORA_ADMIN_KEY` is set, and then every request under `/v1/lora_adapters` needs that bearer token.

## Scope when adapters return

- **Model family:** dense qwen2/llama-shaped models, served through the driver.
- **Target modules:** the seven attention and MLP projections (`q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, `down_proj`). An adapter that targets any other module is refused
  outright at load time; poot does not partially apply an adapter.
- **DoRA:** not supported. A DoRA checkpoint (LoRA plus a magnitude vector) is rejected with a clear error
  rather than loaded as a plain LoRA with the magnitude term dropped.

Design notes and primary sources live under
[Architecture: Serving design](../architecture/serving-design.mdx).
