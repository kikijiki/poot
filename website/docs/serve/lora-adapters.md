---
id: lora-adapters
title: Using LoRA adapters
sidebar_position: 11
---

# Using LoRA adapters

Adapters load and unload while the server runs, through `/v1/lora_adapters`, and a request selects one by name.
The driver runs every adapter of one load together as one generation of weights beside the base weights, so a
request that names no adapter runs the base model unchanged.

## Administration routes

| Request                                          | Effect                                                                                                        |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `GET /v1/lora_adapters`                          | Lists the loaded adapters: `{"object": "list", "data": [{"name", "index"}]}`.                                 |
| `POST /v1/lora_adapters` with `{"name", "path"}` | Loads the PEFT adapter directory at `path` (`adapter_config.json` and `adapter_model.safetensors`) as `name`. |
| `DELETE /v1/lora_adapters/{name}`                | Unloads `name`.                                                                                               |

The operations run on the engine thread, which runs them with no request in flight. A load captures every loaded
adapter plus the new one as a new generation and prepares its entries for the shapes the server prepared; the
adapter indices are positions in that set, so unloading one shifts the indices of the ones after it. A request
already running on the replaced generation finishes on it. Statuses: `400` for a name already loaded, an adapter
that does not fit the model or an unreadable directory; `404` for an unknown name; `409` when the last adapter
cannot be unloaded because running requests hold it (retry later); `503` when the prepared-entry limits cannot
take the new generation.

The routes keep their trust policy: on a loopback listener they are open to local processes, on a non-loopback
listener they are disabled unless `--lora-admin-key-file` names a file holding a bearer token, and then every
request under `/v1/lora_adapters` needs that token, on a loopback listener too. The key is read from the file once
at startup, so it never appears in the process arguments. Inference stays unauthenticated.

## Selecting an adapter

A completion or chat request sets `lora_adapter` to a loaded name. A name no loaded adapter has is a `400`, never
a silent fall back to the base model. An adapter unloaded between the request's validation and its start fails
the request with a `400` too.

## Scope

- **Model family:** dense qwen2/llama-shaped models, served through the driver.
- **Target modules:** the seven attention and MLP projections (`q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, `down_proj`). An adapter that targets any other module is refused
  outright at load time; poot does not partially apply an adapter.
- **DoRA:** not supported. A DoRA checkpoint (LoRA plus a magnitude vector) is rejected with a clear error
  rather than loaded as a plain LoRA with the magnitude term dropped.
- **Backends:** verified on wgpu and ROCm.

Design notes and primary sources live under
[Architecture: Serving design](../architecture/serving-design.mdx).
