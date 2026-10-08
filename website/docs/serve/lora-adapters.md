---
id: lora-adapters
title: Using LoRA adapters
sidebar_position: 11
---

# Using LoRA adapters

:::caution[Being restored]
poot is being refactored. This is not available yet and is planned to return.
:::

Today no adapter can be registered: the server has no adapter flags and no administration route that works.

## What the server does today

- There are no adapter options: like any other unknown `--` option, passing one
  fails startup with an `unknown option` error.
- Every `/v1/lora_adapters` route (`GET`, `POST`, `DELETE`) answers a typed `503` once a request passes the
  trust policy below; none lists, registers or unloads an adapter.
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
