---
id: lora-adapters
title: Using LoRA adapters
sidebar_position: 11
---

# Using LoRA adapters

Adapters load and unload while the server runs, through `/v1/lora_adapters`, and a request selects one by name. A request that names no adapter runs the base model unchanged.

## Administration routes

| Request                                          | Effect                                                                                  |
| ------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `GET /v1/lora_adapters`                          | Lists the loaded adapters: `{"object": "list", "data": [{"name", "index"}]}`.          |
| `POST /v1/lora_adapters` with `{"name", "path"}` | Loads the PEFT adapter directory at `path` (`adapter_config.json`, `adapter_model.safetensors`) as `name`. |
| `DELETE /v1/lora_adapters/{name}`                | Unloads `name`.                                                                         |

Operations run on the engine thread with no request in flight. A load makes every loaded adapter plus the new one a new generation of weights. A request already running on the replaced generation finishes on it. Adapter indices are positions in the set, so unloading one shifts the indices after it.

| Status | Cause                                                                                    |
| ------ | ---------------------------------------------------------------------------------------- |
| `400`  | A name already loaded, an adapter that does not fit the model, or an unreadable directory |
| `404`  | An unknown name                                                                          |
| `409`  | The last adapter cannot be unloaded because running requests hold it (retry later)      |
| `503`  | The prepared-entry limits cannot take the new generation                                 |

On a loopback listener the routes are open to local processes. On a non-loopback listener they are disabled unless `--lora-admin-key-file` names a file holding a bearer token. Then every request under `/v1/lora_adapters` needs that token, on loopback too. The key is read once at startup and never appears in the process arguments. Inference stays unauthenticated.

## Selecting an adapter

A completion or chat request sets `lora_adapter` to a loaded name. An unknown name is a `400`, never a fall back to the base model. So is an adapter unloaded between validation and start.

## Scope

- **Models:** dense decoders whose attention and MLP projections carry the module names below; see [Models](../develop/models.md).
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. An adapter that targets any other module is refused at load; poot does not partially apply an adapter.
- **DoRA:** not supported; a DoRA checkpoint is rejected with an error.
- **Dtype:** F32, F16 or BF16 factors, uploaded as stored. The adapters of one load must share one stored dtype; a mixed set is refused.
- **Backends:** device tests cover wgpu, ROCm and PTX; none covers adapters on raw Vulkan.

Design: [Serving design](../architecture/serving-design.mdx).
