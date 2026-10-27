---
id: tensor-parallelism
title: Tensor parallelism
sidebar_position: 12
---

# Tensor parallelism

`--placement tensor:ORD,ORD,...` asks for one tensor-parallel rank per GPU ordinal, on the `ptx` backend only; a group on another backend is refused at startup. `--placement pipeline:ORD,ORD,...` splits the layers across GPUs instead. No environment variable selects either.

## What a rank holds

- Weights are divided by role (Q, K, V, gate and up by output; O and down by input), and every rank loads only its share.
- The paged KV pool is divided by KV heads, so each rank holds its own heads of every pool.
- The memory fit checks each rank against its own GPU, including its pool and state bytes. A short rank stops startup naming it.
- A head count that does not divide over the ranks is refused at startup by name.

## What is verified

Verified on NVIDIA GPUs joined by NVLink:

- Tensor groups and layer pipelines served requests on two GPUs of one node. Greedy requests matched the single-GPU result to rounding, and sampled requests matched the single-GPU sampler's tokens.
- A tensor group of four GPUs decoded a tiny model to the single-GPU logits, every rank summing the partials of all four.

Not covered: serving a group of four, a real checkpoint, and GPUs that reach each other only across CPU sockets (on one such host peer copies delivered no data, so check yours first).

Design: [Serving design](../architecture/serving-design.mdx).
